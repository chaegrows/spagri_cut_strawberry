#include <iostream>
#include <rclcpp/rclcpp.hpp>

#include <md200_lift1400_def.hpp>
#include <yaml-cpp/yaml.h>


using std::cout;
using std::endl;

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  std::shared_ptr<mf_serial::Serial_Lift1400> ptr =
    std::make_shared<mf_serial::Serial_Lift1400>("lift1400_serial");


  // parse config
  YAML::Node config =
    YAML::LoadFile("/opt/config/common/serial_info.yaml");
  auto lift_config = config["lift1400"];
  if (!lift_config){
    cout << "lift1400 config not found" << endl;
    return 1;
  }

  // initialized serial port
  std::string vid = lift_config["vendorid"].as<std::string>();
  std::string pid = lift_config["productid"].as<std::string>();
  int baudrate = lift_config["baudrate"].as<int>();
  bool verbose = lift_config["verbose"].as<bool>();
  bool reconnect = lift_config["reconnect"].as<bool>();
  int reconnect_delay = lift_config["reconnect_delay_ms"].as<int>();

  cout << "vid: " << vid << endl;
  cout << "pid: " << pid << endl;
  cout << "baudrate: " << baudrate << endl;
  cout << "verbose: " << verbose << endl;
  cout << "reconnect: " << reconnect << endl;
  cout << "reconnect_delay: " << reconnect_delay << endl;

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
