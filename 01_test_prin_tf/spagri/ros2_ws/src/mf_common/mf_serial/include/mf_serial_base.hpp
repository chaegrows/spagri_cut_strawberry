#pragma once
//try to write header-only library...

#include "serial_driver/serial_port.hpp"
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <deque>

using drivers::serial_driver::SerialPortConfig;
using drivers::serial_driver::SerialPort;

namespace mf_serial{
struct MfSerialConfig{
public:
  MfSerialConfig()
    :verbose(false), baud_rate(0), vendorid(""), productid(""), device(""), _port(""),
    enable_reconnect(false), reconnect_delay(0)
  {}
  bool verbose;
  uint32_t baud_rate;
  std::string vendorid, productid, _port; // hex string
  std::string device;        ///< user-given tty path (optional)
  bool enable_reconnect;
  size_t reconnect_delay; // milliseconds

  // Modifying below things are not implemented
  // flowcontrol
  // parity
  // stopbits
  std::string to_string() const
  {
    return "MfSerialConfig{ baud=" + std::to_string(baud_rate) +
           ", vid=" + vendorid + ", pid=" + productid +
           ", device=" + device +
           ", verbose=" + std::to_string(verbose) +
           ", reconnect=" + std::to_string(enable_reconnect) +
           ", delay=" + std::to_string(reconnect_delay) + " }";
  }

public: // helpers
  bool is_ok() const
  {
    bool ok = baud_rate > 0;
    bool have_vidpid = !vendorid.empty() && !productid.empty();
    bool have_dev    = !device.empty();
    ok &= (have_vidpid || have_dev);
    if (enable_reconnect) ok &= (reconnect_delay > 0);
    return ok;
  }
};

enum {
  SERIAL_NOT_READY = 0,
  SERIAL_READY = 1,
  SERIAL_RUNNING = 2
};

class MfSerialBase: public rclcpp::Node{
public: // these must be defined in derived class
  virtual bool is_rx_msg_valid(const std::deque<uint8_t>& data, int& start_index, int& end_idx) {
    std::cout << "implement is_rx_msg_valid()!!" << std::endl;
    (void)data; (void)start_index; (void)end_idx; return false;}; //called to check if RX msg valid
  virtual void read_callback_on_success(const std::deque<uint8_t>& data) {
    std::cout << "implement read_callback_on_success()!!" << std::endl;
    (void)data;}; // called when RX msg is valid
public:
  struct SerialState{
    int state;
    std::mutex mtx;
    int getState(){
      return state;
    }
    void setState(int new_state){
      mtx.try_lock();
      state = new_state;
    }
  };
public:
  MfSerialBase(std::string node_name);
  ~MfSerialBase();
  bool connect();
  void init_write_thread(size_t interval_ms);
  void init_read_thread();
  void set_write_buffer(const std::vector<uint8_t>& data);

  // serial thread
  void write_timer_callback(size_t interval_ms); //loop
  void read_timer_callback(); //loop
  void _force_write(const std::vector<uint8_t>& data);

  // set functions
  MfSerialBase& setVID_PID(std::string vid, std::string pid){
    mf_serial_config.vendorid = vid; mf_serial_config.productid = pid; return *this;}
  MfSerialBase& setBaudrate(int baud_rate){
    mf_serial_config.baud_rate = baud_rate; return *this;}
  MfSerialBase& setDevice(std::string device){
    mf_serial_config.device = device; return *this;}
  MfSerialBase& setVerbose(bool verbose){
    mf_serial_config.verbose = verbose; return *this;}
  MfSerialBase& setReconnect(bool enable_reconnect, size_t reconnect_delay=0){
    mf_serial_config.enable_reconnect = enable_reconnect;
    mf_serial_config.reconnect_delay = reconnect_delay; return *this;}
  bool is_config_ok() const {return mf_serial_config.is_ok();}

  // get functions
  std::string getVID_PID() const {
    return mf_serial_config.vendorid + ":" +
            mf_serial_config.productid;}
  std::string getLastErrorMsg() const {return last_error_msg;}
  
  bool getVerbose() const {
      return mf_serial_config.verbose;
  }
private: //internally called
  bool _connect();
  void setLastErrorMsg(std::string msg);
  std::deque<uint8_t> _squeeze_and_get_data(std::deque<uint8_t>& read_buffer, int start_idx, int end_idx);
  
private:
  std::shared_ptr<SerialPort> _serial_port_ptr;
  MfSerialConfig mf_serial_config;
  IoContext ctx;
  std::string last_error_msg;
  SerialState connect_state;
  std::mutex write_mtx;

  

  // serial IO
  std::vector<uint8_t> write_buffer;
  std::deque<uint8_t> read_buffer;
  std::vector<std::thread> serial_threads;
};

} // namespace mf_serial
