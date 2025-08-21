#include <iostream>
#include <rclcpp/rclcpp.hpp>

#include <md200_lift1400_def.hpp>
#include <yaml-cpp/yaml.h>


using std::cout;
using std::endl;

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  std::shared_ptr<mf_serial::Serial_MD200> ptr =
    std::make_shared<mf_serial::Serial_MD200>("md200_serial");


  // parse config
  YAML::Node config =
    YAML::LoadFile("/opt/config/common/serial_info.yaml");
  auto md_config = config["md200"];
  if (!md_config){
    cout << "md200 config not found" << endl;
    return 1;
  }

  // initialized serial port
  std::string vid = md_config["vendorid"].as<std::string>();
  std::string pid = md_config["productid"].as<std::string>();
  int baudrate = md_config["baudrate"].as<int>();
  bool verbose = md_config["verbose"].as<bool>();
  bool reconnect = md_config["reconnect"].as<bool>();
  int reconnect_delay = md_config["reconnect_delay_ms"].as<int>();


  ptr->setVID_PID(vid, pid)
    .setBaudrate(baudrate)
    .setVerbose(verbose) // enable debugging
    .setReconnect(reconnect, reconnect_delay)
    ;

  if (ptr->is_config_ok()){
    bool connected = ptr->connect();
    cout << "connected? " << connected << endl;

    ptr->init_read_thread();
    ptr->init_write_thread(100);
    cout << "serial port activated" << endl;
    rclcpp::spin(ptr);
  }
  else{
    cout << "config is not ok" << endl;
  }

  ptr.reset();
  cout << "program terminated successfully" << endl;
}
