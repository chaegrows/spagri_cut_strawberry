#pragma once

#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <mf_serial_base.hpp>
#include <mf_msgs/msg/endeffector.hpp>

////////////////////////////////////////////
// 241130
// start1         uint8
// start2         uint8
// sender_dev_id  uint16
// sw_ver         uint16
// len_frame      uint16
// n_position     uint8
// positions      uint16 * 10
// n_digital_ios  uint8
// digital_ios    uint16 * 10
// event          uint32
// crc            uint8
// end            uint8

const int WRITE_THREAD_LATENCY = 100;



namespace mf_serial{

constexpr static int MF_FRAME_LENGTH = 56;
constexpr static int N_POSITION_MAX = 10;
constexpr static int N_DIGITAL_IO_MAX = 10;

using POSITION_TYPE = uint16_t;
using DIGITAL_IO_TYPE = uint16_t;

std::string uint8ToHexString(uint8_t value) {
  std::ostringstream oss;
  oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(value);
  return oss.str();
}

void init_mf_ros_msg(mf_msgs::msg::Endeffector& data){
  data.sw_version = 0;
  data.len_frame = MF_FRAME_LENGTH;
  data.n_position = 0;
  for (int i = 0; i < N_POSITION_MAX; ++i){
    data.positions.push_back(0);
  }
  data.n_digital_io = 0;
  for (int i = 0; i < N_DIGITAL_IO_MAX; ++i){
    data.digital_ios.push_back(0);
  }
  data.event = mf_msgs::msg::Endeffector::EVENT_NONE;
}

bool is_mf_rx_msg_valid(const std::deque<uint8_t>& data, int& start_idx, int& end_idx, std::string& err_str){
  // min length check
  if (data.size() < MF_FRAME_LENGTH) {
    err_str = "frame is shorter than min length";
    return false;
  }

  // start1, start2, end check
  constexpr static int START1_VALUE = 0x7E;
  constexpr static int START2_VALUE = 0xAA;
  constexpr static int END_VALUE = 0x03;

  // for (size_t st = 0; st < data.size() - MF_FRAME_LENGTH - 1; ++st){
  for (int st = int(data.size() - MF_FRAME_LENGTH); st >= 0; --st){
    if (data[st]  != START1_VALUE) continue;
    if (data[st+1]   != START2_VALUE) continue;

    // length
    uint16_t length_received_data;
    std::memcpy(&length_received_data, &data[st+6], sizeof(length_received_data));

    // end check
    // std::cout << "length received data: " << length_received_data << std::endl;
    if (st+length_received_data >= int(data.size())) continue;
    if (data[st+length_received_data - 1] != END_VALUE) continue;

    // checksum check - xor checksum from 0 to end - 2
    uint8_t checksum = 0;
    for (int i = st; i < st + length_received_data - 2; ++i){
      checksum ^= data[i];
    }
    if (checksum != data[st+length_received_data - 2]) continue;

    start_idx = static_cast<int>(st);
    end_idx = st + length_received_data;
    return true;
  }
  return false;
}

void serialize_mf_data(const mf_msgs::msg::Endeffector& data, std::vector<uint8_t>& buffer){
  buffer.resize(0);

  buffer.push_back(0x7E);

  buffer.push_back(0xAA);

  uint16_t dev_id = mf_msgs::msg::Endeffector::DEV_HOST_PC;
  uint8_t* dev_id_ptr = (uint8_t*)&dev_id;
  for (size_t i = 0; i < sizeof(uint16_t); i++){
    buffer.push_back(dev_id_ptr[i]);
  }

  uint8_t* sw_ver_ptr = (uint8_t*)&data.sw_version;
  for (size_t i = 0; i < sizeof(uint16_t); i++){
    buffer.push_back(sw_ver_ptr[i]);
  }
  uint16_t len_frame = MF_FRAME_LENGTH;
  uint8_t* len_frame_ptr = (uint8_t*)&len_frame;
  for (size_t i = 0; i < sizeof(uint16_t); i++){
    buffer.push_back(len_frame_ptr[i]);
  }

  uint8_t n_position = data.n_position;
  uint8_t* n_position_ptr = (uint8_t*)&n_position;
  for (size_t i = 0; i < sizeof(uint8_t); i++){
    buffer.push_back(n_position_ptr[i]);
  }

  for (size_t i = 0 ; i < n_position; ++i){
    float f_data = data.positions[i];
    POSITION_TYPE val = static_cast<POSITION_TYPE>(f_data * 100);
    uint8_t* val_ptr = (uint8_t*)&val;
    for (size_t j = 0; j < sizeof(POSITION_TYPE); j++){
      buffer.push_back(val_ptr[j]);
    }
  }
  size_t remained_positions = N_POSITION_MAX - n_position;
  for (size_t i = 0; i < remained_positions; ++i){
    POSITION_TYPE val = 0;
    uint8_t* val_ptr = (uint8_t*)&val;
    for (size_t j = 0; j < sizeof(POSITION_TYPE); j++){
      buffer.push_back(val_ptr[j]);
    }
  }

  uint8_t n_digital_io = data.n_digital_io;
  uint8_t* n_digital_io_ptr = (uint8_t*)&n_digital_io;
  for (size_t i = 0; i < sizeof(uint8_t); i++){
    buffer.push_back(n_digital_io_ptr[i]);
  }

  for (size_t i = 0 ; i < n_digital_io; ++i){
    DIGITAL_IO_TYPE val = data.digital_ios[i];
    uint8_t* val_ptr = (uint8_t*)&val;
    for (size_t j = 0; j < sizeof(DIGITAL_IO_TYPE); j++){
      buffer.push_back(val_ptr[j]);
    }
  }
  size_t remained_digital_io = N_DIGITAL_IO_MAX - n_digital_io;
  for (size_t i = 0; i < remained_digital_io; ++i){
    DIGITAL_IO_TYPE val = 0;
    uint8_t* val_ptr = (uint8_t*)&val;
    for (size_t j = 0; j < sizeof(DIGITAL_IO_TYPE); j++){
      buffer.push_back(val_ptr[j]);
    }
  }

  uint32_t event = data.event;
  uint8_t* event_ptr = (uint8_t*)&event;
  for (size_t i = 0; i < sizeof(uint32_t); i++){
    buffer.push_back(event_ptr[i]);
  }

  uint8_t xor_checksum = 0;
  for (size_t i = 0; i < buffer.size(); ++i){
    xor_checksum ^= buffer[i];
  }
  buffer.push_back(xor_checksum);

  buffer.push_back(0x03);
}

bool deserialize_mf_data(const std::deque<uint8_t>& buffer, mf_msgs::msg::Endeffector& data, std_msgs::msg::Header header){
  if (buffer.size() != MF_FRAME_LENGTH) return false;

  data.header = header;

  std::memcpy(&data.sw_version, &buffer[4], sizeof(data.sw_version));

  data.len_frame = MF_FRAME_LENGTH;

  std::memcpy(&data.n_position, &buffer[8], sizeof(data.n_position));

  for (int i = 0; i < N_POSITION_MAX; ++i){
    POSITION_TYPE val;
    std::memcpy(&val, &buffer[9+i*2], sizeof(val));
    data.positions[i] = (1.0f*val / 100);
  }

  std::memcpy(&data.n_digital_io, &buffer[29], sizeof(data.n_digital_io));
  for (int i = 0; i < N_DIGITAL_IO_MAX; ++i){
    DIGITAL_IO_TYPE val;
    std::memcpy(&val, &buffer[30+i*2], sizeof(val));
    data.digital_ios[i] = (val);
  }

  uint32_t event;
  std::memcpy(&event, &buffer[50], sizeof(data.event));
  data.event = event;

  return true;
}

class Serial_MfEEF : public MfSerialBase{
  // base class is ROS2 node
public:
  Serial_MfEEF(std::string node_name)
    :MfSerialBase(node_name)
  {
    serial_sub = this->create_subscription<mf_msgs::msg::Endeffector>("mf_eef_raw/in", 10, std::bind(&Serial_MfEEF::topic_callback, this, std::placeholders::_1));
    serial_pub = this->create_publisher<mf_msgs::msg::Endeffector>("mf_eef_raw/out", 10);

    {
    }
  }

  bool is_rx_msg_valid(
      const std::deque<uint8_t>& data, int& start_index, int& end_idx) override
    {
    std::string err_str;
    bool msg_ok = mf_serial::is_mf_rx_msg_valid(data, start_index, end_idx, err_str);
    if (enable_print_raw){
      for (size_t i = 0; i < data.size(); ++i){
        std::cout << uint8ToHexString(data[i]) << ' ';
      }
    }
    return msg_ok;
  }

  void read_callback_on_success(const std::deque<uint8_t>& data) override{
    std_msgs::msg::Header header;
    header.stamp = this->now();
    static mf_msgs::msg::Endeffector msg;
    if (msg.positions.size() == 0) init_mf_ros_msg(msg);

    if (deserialize_mf_data(data, msg, header)){
      serial_pub->publish(msg);
    }
  }

  void topic_callback(const mf_msgs::msg::Endeffector::SharedPtr msg){
      std::vector<uint8_t> buffer;
      serialize_mf_data(*msg, buffer);
      set_write_buffer(buffer);

      if (false == write_thread_enabled){
        this->init_write_thread(WRITE_THREAD_LATENCY);
        write_thread_enabled = true;
      }
    }

  Serial_MfEEF& set_print_raw(bool val){
    enable_print_raw = val;
    return *this;
  }

private:
  rclcpp::Subscription<mf_msgs::msg::Endeffector>::SharedPtr serial_sub;
  rclcpp::Publisher<mf_msgs::msg::Endeffector>::SharedPtr serial_pub;
  bool enable_print_raw = false;
  bool write_thread_enabled = false;
};

class Serial_MfEEF_Emulator : public MfSerialBase{
  // base class is ROS2 node
public:
  Serial_MfEEF_Emulator(std::string node_name)
    :MfSerialBase(node_name)
  {
    mf_msgs::msg::Endeffector msg;
    init_mf_ros_msg(msg);
    msg.sw_version = 0x0001;
    msg.len_frame = MF_FRAME_LENGTH;
    msg.n_position = 0;
    msg.n_digital_io = 0;
    msg.event = mf_msgs::msg::Endeffector::EVENT_NONE;

    std::vector<uint8_t> buffer;
    serialize_mf_data(msg, buffer);
    set_write_buffer(buffer);
  }

private:
  rclcpp::Subscription<mf_msgs::msg::Endeffector>::SharedPtr serial_sub;
  rclcpp::Publisher<mf_msgs::msg::Endeffector>::SharedPtr serial_pub;

};

}//namespace mf_serial
