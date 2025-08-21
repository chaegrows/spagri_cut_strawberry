#include "mf_serial_base.hpp"
#include <iostream>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <deque>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cctype> 

namespace mf_serial{


using std::cout;
using std::endl;
constexpr static int MAX_SERIAL_N_READ_BYTES = 1024;
constexpr static int MAX_SERIAL_READ_BUFFER_LENGTH = 1024;
namespace fs = std::filesystem;

std::vector<std::string> findUsbPort(const std::string& targetVid, const std::string& targetPid) {
  auto t_vid = targetVid; 
  auto t_pid = targetPid; 
  std::transform(t_vid.begin(), t_vid.end(), t_vid.begin(), tolower);
  std::transform(t_pid.begin(), t_pid.end(), t_pid.begin(), tolower);

  std::string basePath = "/sys/bus/usb/devices";

  std::vector<std::string> ports;
  for (const auto& entry : fs::directory_iterator(basePath)) {
    if (fs::is_directory(entry)) {
      std::string devicePath = entry.path();
      std::string vidPath = devicePath + "/idVendor";
      std::string pidPath = devicePath + "/idProduct";

      if (fs::exists(vidPath) && fs::exists(pidPath)) {
        std::ifstream vidFile(vidPath);
        std::ifstream pidFile(pidPath);

        std::string vid, pid;
        std::getline(vidFile, vid);
        std::getline(pidFile, pid);

        if (vid == t_vid && pid == t_pid) {
          std::string portPath = devicePath + ":1.0";
          for (auto port : fs::directory_iterator(portPath)) {
            auto str = port.path().filename().string();
            if (str.size() > 3 && str.substr(0, 3) == "tty") {
              ports.push_back("/dev/" + port.path().filename().string());
            }
          }
        }
      }
    }
  }

  return ports; // 포트를 찾지 못한 경우 빈 문자열 반환
}


MfSerialBase::MfSerialBase(std::string node_name)
  : Node(node_name)
  , _serial_port_ptr(nullptr), ctx(1) // one allow one thread
  , read_buffer(0)
  {
  }

void MfSerialBase::init_write_thread(size_t interval_ms){
  if (interval_ms == 0) {
    setLastErrorMsg("write_thread_invoke_interval_ms is 0");
    exit(1);
  }
  std::thread t(&MfSerialBase::write_timer_callback, this, interval_ms);
  serial_threads.push_back(std::move(t));
  cout << "write thread initialized with port "
    << getVID_PID() << endl;
}

void MfSerialBase::init_read_thread(){
  std::thread t(&MfSerialBase::read_timer_callback, this);
  serial_threads.push_back(std::move(t));
  cout << "read thread initialized with port "
    << getVID_PID() << endl;
}

void MfSerialBase::write_timer_callback(size_t interval_ms){
  while(rclcpp::ok()){
    auto start = std::chrono::high_resolution_clock::now();
    
    if (connect_state.getState() != SERIAL_READY){
      setLastErrorMsg("serial is not connected in write thread");
      std::this_thread::sleep_for(
        std::chrono::milliseconds(mf_serial_config.reconnect_delay));
      continue;
    }
    
    try{
      if (write_buffer.size() != 0){
        size_t n_wrote = _serial_port_ptr->send(write_buffer);
        if (mf_serial_config.verbose){
          cout << n_wrote << " bytes wrote to " 
            << getVID_PID() << endl;
        }
      }
      else {
        setLastErrorMsg("write_buffer is empty");
      }
      // sleep
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      size_t sleep_time = interval_ms - duration.count();
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    }
    catch (const std::system_error& e){
      continue;
    }
  }
}

void MfSerialBase::read_timer_callback(){
  static std::vector<uint8_t> read_buffer_tmp(MAX_SERIAL_N_READ_BYTES);
  // read thread does not sleep
  while (rclcpp::ok()){
    if (connect_state.getState() != SERIAL_READY){
      setLastErrorMsg("serial is not connected in read thread");
      std::this_thread::sleep_for(
        std::chrono::milliseconds(mf_serial_config.reconnect_delay));
      continue;
    }

    try{
      size_t n_read = _serial_port_ptr->receive(read_buffer_tmp);
      read_buffer.insert(read_buffer.end(), read_buffer_tmp.begin(), read_buffer_tmp.begin() + n_read);
      if (mf_serial_config.verbose){
        cout << "n_read, buffer size: " << n_read << ' ' << read_buffer.size() << endl;
      }
      if (read_buffer.size() >= MAX_SERIAL_READ_BUFFER_LENGTH){ //squeez if data is not decoded for a long time
        read_buffer.erase(read_buffer.begin(), read_buffer.begin() + MAX_SERIAL_READ_BUFFER_LENGTH);
      }
      
      int start_idx, end_idx;
      while (is_rx_msg_valid(read_buffer, start_idx, end_idx)){
        std::deque<uint8_t> valid_data(
          read_buffer.begin() + start_idx,
          read_buffer.begin() + end_idx
        );
        read_buffer.erase(read_buffer.begin(), read_buffer.begin() + end_idx);
        
        if (mf_serial_config.verbose){
          std::cout << "valid_data: " << valid_data.size() << std::endl;
          for (size_t i = 0; i < valid_data.size(); ++i){
            std::cout << "valid_data[" << i << "]: " << (int)valid_data[i] << std::endl;
          }
        }

        read_callback_on_success(valid_data);
      }
    }
    catch (const std::system_error& e){
      continue;
    }
  }
}

std::deque<uint8_t> MfSerialBase::_squeeze_and_get_data(
  std::deque<uint8_t>& read_buffer,
  int start_idx,
  int end_idx)
{
  std::deque<uint8_t> valid_data(
    read_buffer.begin() + start_idx,
    read_buffer.begin() + end_idx
  );
  read_buffer.erase(read_buffer.begin(), read_buffer.begin() + end_idx);
  return valid_data;
}

bool MfSerialBase::_connect()
{
  std::cout << "connect" << std::endl;
  if (connect_state.getState() == SERIAL_READY) return true;
  else 
  {
    std::cout << "connect_state.getState() == SERIAL_NOT_READY" << std::endl;
    connect_state.setState(SERIAL_NOT_READY);
  }

  if (!mf_serial_config.is_ok()) {
    std::cout << "mf_serial_config.is_ok() == false" << std::endl;
    setLastErrorMsg("MfSerialConfig is not ok: " + mf_serial_config.to_string());
    return false;
  }
  std::cout << "mf_serial_config.is_ok()" << std::endl;
  SerialPortConfig cfg(
      mf_serial_config.baud_rate,
      drivers::serial_driver::FlowControl::NONE,
      drivers::serial_driver::Parity::NONE,
      drivers::serial_driver::StopBits::ONE);

  std::vector<std::string> ports;
  std::cout << "mf_serial_config.vendorid.empty() && !mf_serial_config.productid.empty()" << std::endl;
  if ((mf_serial_config.vendorid == "1A86") && (mf_serial_config.productid == "7523")){
    mf_serial_config._port = "/dev/ttyCH341USB0";
  }
  else{
    if (!mf_serial_config.vendorid.empty() && !mf_serial_config.productid.empty()) {
      ports = findUsbPort(mf_serial_config.vendorid, mf_serial_config.productid);
    }
    else if (!mf_serial_config.device.empty()) {        // <-- device 멤버를 사용
      std::cout << "Using device: " << mf_serial_config.device << std::endl;
      ports.push_back(mf_serial_config.device);
    }
    else {
      for (const auto& entry : fs::directory_iterator("/dev")) {
        auto name = entry.path().filename().string();
        if (name.rfind("ttyUSB", 0) == 0 || name.rfind("ttyACM", 0) == 0)
          ports.emplace_back("/dev/" + name);
      }
    }

    if (ports.empty()) {
      setLastErrorMsg("No serial port found (check VID/PID or device)");
      return false;
    }

    mf_serial_config._port = ports.front();
  }
  cout << "Using serial port: " << mf_serial_config._port << endl;

  _serial_port_ptr = std::make_unique<SerialPort>(ctx, mf_serial_config._port, cfg);

  try {
    _serial_port_ptr->open();
    connect_state.setState(SERIAL_READY);
    return true;
  } catch (const std::system_error& e) {
    setLastErrorMsg("Failed to open " + mf_serial_config._port + ": " + e.what());
    return false;
  }
}


bool MfSerialBase::connect()
{
  std::lock_guard<std::mutex> lg(connect_state.mtx);
  if (connect_state.getState() == SERIAL_READY) return true;

  if (!mf_serial_config.enable_reconnect) return _connect();

  while (rclcpp::ok()) {
    if (_connect()) return true;
    std::this_thread::sleep_for(
        std::chrono::milliseconds(mf_serial_config.reconnect_delay));
  }
  return false;
}

void MfSerialBase::setLastErrorMsg(std::string msg)
{
  last_error_msg = std::move(msg);
  if (mf_serial_config.verbose) cout << last_error_msg << endl;
}

void MfSerialBase::set_write_buffer(const std::vector<uint8_t>& data){
  std::lock_guard<std::mutex> lock(write_mtx);
  write_buffer = data;
}

void MfSerialBase::_force_write(const std::vector<uint8_t>& data){
  size_t n_wrote = _serial_port_ptr->send(data);
  if (mf_serial_config.verbose)
    cout << n_wrote << " bytes wrote to " << getVID_PID() << endl;
}


MfSerialBase::~MfSerialBase()
{
  if (_serial_port_ptr && _serial_port_ptr->is_open()) {
    _serial_port_ptr->close();
    connect_state.setState(SERIAL_NOT_READY);
    cout << "Serial port " << _serial_port_ptr->device_name() << " closed" << endl;
  }
}

} // namespace mf_serial
