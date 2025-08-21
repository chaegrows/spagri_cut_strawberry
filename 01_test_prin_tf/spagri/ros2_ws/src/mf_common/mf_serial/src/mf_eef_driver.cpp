#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

#include <mf_frame.hpp>

using std::cout;
using std::endl;

int main(int argc, char **argv)
{

  rclcpp::init(argc, argv);


  auto driver = std::make_shared<mf_serial::Serial_MfEEF>(
                  "end_effector_driver_node");

  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile("/opt/config/common/serial_info.yaml");
    std::cout << "YAML load success" << std::endl;
  } catch (const YAML::BadFile &e) {
    std::cerr << "YAML load error: " << e.what() << endl;
    return 1;
  }

  YAML::Node eef_cfg = cfg["end_effector"];
  if (!eef_cfg) {
    std::cerr << "end_effector section not found in yaml" << endl;
    return 1;
  }

  // std::string device          = eef_cfg["device"].as<std::string>("");
  // initialized serial port
  std::string vid = eef_cfg["vendorid"].as<std::string>();
  std::string pid = eef_cfg["productid"].as<std::string>();
  uint32_t    baudrate        = eef_cfg["baudrate"].as<uint32_t>(0);
  bool        verbose         = eef_cfg["verbose"].as<bool>(false);
  bool        reconnect_en    = eef_cfg["reconnect"].as<bool>(false);
  int         reconnect_delay = eef_cfg["reconnect_delay_ms"].as<int>(0);
  
  driver->setVID_PID(vid, pid)
  // driver->setDevice(device)
        .setBaudrate(baudrate)
        .setVerbose(verbose)
        .setReconnect(reconnect_en, reconnect_delay);

  // std::cout << "Serial config: " << device << std::endl;
  if (driver->is_config_ok()){
    std::cout << "Serial config is ok" << std::endl;
    bool connected = driver->connect();
    if (!connected)
      std::cerr << "connect failed: " << driver->getLastErrorMsg() << std::endl;
    else
      std::cout << "connected? " << connected << std::endl;
    driver->init_read_thread();
    std::cout << "serial port activated" << std::endl;
  }
  else{
    std::cout << "config is not ok" << std::endl;
  }


  rclcpp::spin(driver);

  cout << "program terminated successfully" << endl;
}
