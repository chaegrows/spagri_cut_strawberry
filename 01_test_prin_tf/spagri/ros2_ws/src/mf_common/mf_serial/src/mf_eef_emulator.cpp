#include <iostream>
#include <rclcpp/rclcpp.hpp>

#include <mf_frame.hpp>

using std::cout;
using std::endl;

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  std::shared_ptr<mf_serial::Serial_MfEEF_Emulator> ptr = 
    std::make_shared<mf_serial::Serial_MfEEF_Emulator>("strawberry_pollinator_emulator");
  auto& ser = *ptr;

  ser.setPort("/dev/ttyUSB1")
    .setBaudrate(115200)
    .setVerbose(true) // enable debugging
    .setReconnect(true, 1000);
      
  if (ser.is_config_ok()){
    bool connected = ser.connect();
    cout << "connected? " << connected << endl;
    ser.init_write_thread(100);
    cout << "serial port activated" << endl;
  }
  else{
    cout << "config is not ok" << endl;
  }

  rclcpp::spin(ptr);  

  cout << "program terminated successfully" << endl;
}