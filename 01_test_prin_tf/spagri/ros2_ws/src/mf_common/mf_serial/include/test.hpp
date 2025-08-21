#pragma once
// header-only base library for mf_serial

#include "serial_driver/serial_port.hpp"
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <deque>

using drivers::serial_driver::SerialPortConfig;
using drivers::serial_driver::SerialPort;

namespace mf_serial{
struct MfSerialConfig
{
  MfSerialConfig()
  :verbose(false), baud_rate(0) , vendorid("") , productid("") , device("") , _port(""),
   enable_reconnect(false), reconnect_delay(0)
  {}
  bool        verbose;
  uint32_t    baud_rate;     ///< e.g. 115200
  std::string vendorid;      ///< hex string (optional)
  std::string productid;     ///< hex string (optional)
  std::string device;        ///< user-given tty path (optional)
  std::string _port;         ///< selected port (internal)
  
  bool     enable_reconnect;
  size_t   reconnect_delay;  ///< msec

  /* ---- helpers ---- */
  std::string to_string() const
  {
    return "MfSerialConfig{ baud=" + std::to_string(baud_rate) +
           ", vid=" + vendorid + ", pid=" + productid +
           ", device=" + device +
           ", verbose=" + std::to_string(verbose) +
           ", reconnect=" + std::to_string(enable_reconnect) +
           ", delay=" + std::to_string(reconnect_delay) + " }";
  }
  /// valid when baud_rate>0 AND ( (vid&&pid) || device )
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

/*----------------------------------------------------
 *  Serial state enum
 *---------------------------------------------------*/
enum { SERIAL_NOT_READY = 0, SERIAL_READY = 1, SERIAL_RUNNING = 2 };

/*----------------------------------------------------
 *  MfSerialBase  (abstract)
 *---------------------------------------------------*/
class MfSerialBase : public rclcpp::Node
{
public:
  /* ---- to be implemented in derived class ---- */
  virtual bool is_rx_msg_valid(const std::deque<uint8_t>&, int&, int&)
  { std::cout << "implement is_rx_msg_valid()!!\n"; return false; }

  virtual void read_callback_on_success(const std::deque<uint8_t>&)
  { std::cout << "implement read_callback_on_success()!!\n"; }

  /* ---- ctor / dtor ---- */
  explicit MfSerialBase(std::string node_name);
  ~MfSerialBase();

  /* ---- connection ---- */
  bool connect();

  /* ---- thread helpers ---- */
  void init_write_thread(size_t interval_ms);
  void init_read_thread();

  /* ---- write helpers ---- */
  void set_write_buffer(const std::vector<uint8_t>& data);
  void _force_write(const std::vector<uint8_t>& data);

  /* ---- configuration setters ---- */
  MfSerialBase& setVID_PID(const std::string& vid, const std::string& pid)
  { mf_serial_config.vendorid = vid; mf_serial_config.productid = pid; return *this; }

  MfSerialBase& setDevice(const std::string& dev)
  { mf_serial_config.device = dev; return *this; }

  MfSerialBase& setBaudrate(uint32_t br)
  { mf_serial_config.baud_rate = br; return *this; }

  MfSerialBase& setVerbose(bool v)
  { mf_serial_config.verbose = v; return *this; }

  MfSerialBase& setReconnect(bool en, size_t delay_ms=0)
  { mf_serial_config.enable_reconnect = en; mf_serial_config.reconnect_delay = delay_ms; return *this; }

  bool is_config_ok() const { return mf_serial_config.is_ok(); }

  /* ---- getters ---- */
  std::string getVID_PID() const { return mf_serial_config.vendorid + ":" + mf_serial_config.productid; }
  std::string getLastErrorMsg() const { return last_error_msg; }
  bool        getVerbose() const { return mf_serial_config.verbose; }

protected:                /* internal helpers */
  bool _connect();
  void setLastErrorMsg(std::string msg);

  /* squeeze util */
  std::deque<uint8_t> _squeeze_and_get_data(std::deque<uint8_t>& buf, int start, int end);

  /* thread loops */
  void write_timer_callback(size_t interval_ms);
  void read_timer_callback();

protected:
  /* ---- members ---- */
  std::shared_ptr<SerialPort>   _serial_port_ptr;
  MfSerialConfig                mf_serial_config;
  IoContext                     ctx;
  std::string                   last_error_msg;

  struct SerialState {
    int state{SERIAL_NOT_READY};
    std::mutex mtx;
    int  getState()       { return state; }
    void setState(int st) { std::lock_guard<std::mutex> lk(mtx); state = st; }
  } connect_state;

  std::mutex                  write_mtx;
  std::vector<uint8_t>        write_buffer;
  std::deque<uint8_t>         read_buffer;
  std::vector<std::thread>    serial_threads;
};

}  // namespace mf_serial
