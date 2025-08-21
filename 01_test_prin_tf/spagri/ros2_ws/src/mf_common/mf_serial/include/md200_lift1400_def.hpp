#include <mf_msgs/msg/md200_control.hpp>
#include <mf_msgs/msg/md200_status.hpp>
#include <mf_msgs/msg/lift_control.hpp>
#include <mf_msgs/msg/lift_status.hpp>
#include <deque>
#include <mf_serial_base.hpp>


namespace mf_serial{

// Mobile
constexpr static uint8_t RMID_PC_TO_MD = 184;
constexpr static uint8_t TMID_PC_TO_MD = 172;

constexpr static uint8_t RMID_MD_TO_PC = 172;
constexpr static uint8_t TMID_MD_TO_PC = 184;

constexpr static uint8_t TARGET_PID = 253;
constexpr static size_t MD_FRAME_LENGTH = 26;
constexpr static uint8_t TARGET_ID = 1;
constexpr static uint8_t TARGET_PID_SERVO = 252;

// Lift
constexpr static uint8_t RMID_PC_TO_LIFT = 183;
constexpr static uint8_t TMID_PC_TO_LIFT = 184;

constexpr static uint8_t RMID_LIFT_TO_PC = 184;
constexpr static uint8_t TMID_LIFT_TO_PC = 183;

constexpr static uint8_t TARGET_ID_LIFT = 1; // default

enum class Lift_TARGET_PID{
  CMD_ESTOP = 10,
  CMD_MOVE_HOME = 1,
  CMD_MOVE_ABS = 2,
  CMD_MOVE_REL = 3,
};

constexpr static size_t LIFT_FRAME_LENGTH = 23;

constexpr static double SCALE_TO_MM = 10.0;
constexpr static bool VERBOSE = false;
constexpr static int32_t MIN_HEIGHT = 0;
constexpr static int32_t MAX_HEIGHT = 14000;
/*
  RMID: uint8_t
  TMID: uint8_t
  ID number: uint8_t
  Parameter ID: uint8_t
  Data Number: uint8_t
  Data: uint8_t
  Checksum: uint8_t
*/

struct RxPacket{
  uint8_t rmid;
  uint8_t tmid;
  uint8_t id_number;
  uint8_t parameter_id;
  uint8_t data_number;

  uint32_t nouse1;
  uint32_t nouse2;
  uint16_t nouse3;
  uint8_t bat_percent;
  uint32_t nouse4;
  uint8_t robot_status;
  int16_t linear_speed;
  int16_t angular_speed;

  uint8_t checksum;
}__attribute__((packed));

struct TxServoPacket{
  uint8_t rmid;
  uint8_t tmid;
  uint8_t id_number;
  uint8_t parameter_id;
  uint8_t data_number;
  uint8_t control_mode;
  uint16_t linear_speed;
  uint16_t angular_speed;
  uint8_t reset_odometry;
  uint8_t checksum;
}__attribute__((packed));

uint8_t get_check_sum_md200(uint8_t size, const std::vector<uint8_t>& byArray){
  uint8_t sum = 0;
  for (uint8_t i = 0; i < size; ++i){
    sum += byArray[i];
  }
  uint8_t chk = (~sum) + 1;
  return chk;
}

std::vector<uint8_t> create_tx_packet_lift(
    uint32_t   cmd_type,
    int32_t    height_mm = 0,
    uint8_t    speed     = 88,
    uint8_t    accel     = 13)
{
  using Cmd = mf_msgs::msg::LiftControl;

  // 1) cmd_type 유효성 검사
  if (cmd_type < Cmd::CMD_ESTOP || cmd_type > Cmd::CMD_MOVE_REL) {
    throw std::invalid_argument("Invalid cmd_type");
  }
  size_t idx = static_cast<size_t>(cmd_type);

  // 2) PID, LEN, HOME/ESTOP 데이터 매핑 테이블
  //  PID: 0(ESTOP)=10, 1(HOME)=10, 2(ABS)=219, 3(REL)=220
  //  LEN: 0,1 → 1바이트, 2,3 → 6바이트
  //  CMD_VALUE: ESTOP→4, HOME→90
  static constexpr std::array<uint8_t,4> PID       = {{ 10,  10, 219, 220 }};
  static constexpr std::array<uint8_t,4> LEN       = {{  1,   1,   6,   6  }};
  static constexpr std::array<uint8_t,2> CMD_VALUE = {{ 4,   90        }};

  std::vector<uint8_t> packet;
  packet.reserve(5 + LEN[idx] + 1);  // header(5) + payload + checksum
  packet.push_back(RMID_PC_TO_LIFT);
  packet.push_back(TMID_PC_TO_LIFT);
  packet.push_back(TARGET_ID_LIFT);
  packet.push_back(PID[idx]);
  packet.push_back(LEN[idx]);

  if (idx < 2) {packet.push_back(CMD_VALUE[idx]);}
  else {
    if (cmd_type == Cmd::CMD_MOVE_ABS) {
      int32_t h = std::clamp(static_cast<int32_t>(height_mm * SCALE_TO_MM), MIN_HEIGHT, MAX_HEIGHT);
      packet.push_back(static_cast<uint8_t>( h        & 0xFF));
      packet.push_back(static_cast<uint8_t>((h >>  8) & 0xFF));
      packet.push_back(static_cast<uint8_t>((h >> 16) & 0xFF));
      packet.push_back(static_cast<uint8_t>((h >> 24) & 0xFF));
    } else {
      int32_t rel = static_cast<int32_t>(height_mm * SCALE_TO_MM);
      packet.push_back(static_cast<uint8_t>( rel        & 0xFF));
      packet.push_back(static_cast<uint8_t>((rel >>  8) & 0xFF));
      packet.push_back(static_cast<uint8_t>((rel >> 16) & 0xFF));
      packet.push_back(static_cast<uint8_t>((rel >> 24) & 0xFF));
    }
    packet.push_back(speed);
    packet.push_back(accel);
  }

  // 5) 체크섬 추가
  packet.push_back(get_check_sum_md200(packet.size(), packet));

  return packet;
}


std::vector<uint8_t> create_tx_packet(
  bool enable_estop,
  bool do_free_wheel,
  uint16_t linear_speed = 0,
  uint16_t angular_speed = 0){

  TxServoPacket p;
  p.rmid = RMID_PC_TO_MD;
  p.tmid = TMID_PC_TO_MD;
  p.id_number = TARGET_ID;
  p.parameter_id = TARGET_PID_SERVO;
  p.data_number = 6;
  if (enable_estop){
    p.control_mode = 1;
    p.linear_speed = 0;
    p.angular_speed = 0;
  }
  else if (do_free_wheel){
    p.control_mode = 0;
  }
  else {
    p.control_mode = 1;
    p.linear_speed = linear_speed;
    p.angular_speed = angular_speed;
  }
  p.reset_odometry = 0;

  std::vector<uint8_t> data;
  uint8_t* p_data = (uint8_t*)&p;
  for (size_t i = 0; i < sizeof(TxServoPacket) - 1; ++i){
    data.push_back(p_data[i]);
  }
  uint8_t checksum = get_check_sum_md200(sizeof(TxServoPacket) - 1, data);
  data.push_back(checksum);
  return data;
}


class Serial_MD200: public MfSerialBase{
  // base class is ROS2 node
public:
  Serial_MD200(std::string node_name)
    :MfSerialBase(node_name)
  {
    serial_sub = this->create_subscription<mf_msgs::msg::MD200Control>("md200/in", 1, std::bind(&Serial_MD200::topic_callback, this, std::placeholders::_1));
    serial_pub = this->create_publisher<mf_msgs::msg::MD200Status>("md200/out", 1);

    std::vector<uint8_t> init_data;
    init_data.push_back(184);
    init_data.push_back(172);
    init_data.push_back(1);
    init_data.push_back(4);
    init_data.push_back(1);
    init_data.push_back(253);
    uint8_t check_sum = get_check_sum_md200(init_data.size(), init_data);
    init_data.push_back(check_sum);
    set_write_buffer(init_data);
  }

  ~Serial_MD200(){
    serial_sub = nullptr;
    std::vector<uint8_t> free_wheel_data = create_tx_packet(false, true);
    for (size_t i = 0 ; i < 10; ++i){
      this->_force_write(free_wheel_data);
      usleep(100000);
    }

    std::cout << "Serial_MD200 is destroyed" << std::endl;
  }

  void topic_callback(const mf_msgs::msg::MD200Control::SharedPtr msg){
    int16_t linear_velocity_mm_s = static_cast<int16_t>(msg->twist.linear.x * 1000);
    linear_velocity_mm_s = std::max(
      int16_t(-32768),
      std::min(int16_t(32767), linear_velocity_mm_s));
    int16_t angular_velocity_mm_s = static_cast<int16_t>(msg->twist.angular.z * 1000);
    angular_velocity_mm_s = std::max(
      int16_t(-32768),
      std::min(int16_t(32767), angular_velocity_mm_s));

    std::vector<uint8_t> data = create_tx_packet(
      msg->enable_estop,
      msg->do_free_wheel,
      linear_velocity_mm_s,
      angular_velocity_mm_s);
    set_write_buffer(data);
  }


  bool is_rx_msg_valid(
      const std::deque<uint8_t>& data, int& start_idx, int& end_idx) override
  {
    if (data.size() < MD_FRAME_LENGTH) {
      return false;
    }

    for (size_t st = 0; st < data.size() - MD_FRAME_LENGTH; ++st){
      if (data[st]  != RMID_MD_TO_PC) continue;
      if (data[st+1]!= TMID_MD_TO_PC) continue;

      // length
      uint8_t length_received_data;
      std::memcpy(&length_received_data, &data[st+4], sizeof(length_received_data));
      size_t total_frame_length = length_received_data + 6;

      // checksum check - xor checksum from 0 to end - 2
      std::vector<uint8_t> data_tmp(
        data.begin() + st,
        data.begin() + st + total_frame_length - 1);
      uint8_t checksum = get_check_sum_md200(data_tmp.size(), data_tmp);
      if (checksum != data[st+total_frame_length - 1]) continue;

      start_idx = static_cast<int>(st);
      end_idx = st + length_received_data;
      return true;
    }
    return false;
  }

  void read_callback_on_success(const std::deque<uint8_t>& data) override{
    uint8_t data_array[26];
    std::memcpy(data_array, &data[0], 26);
    RxPacket* packet = (RxPacket*)data_array;

    // bool is_estop_enabled = packet->robot_status & 0x01;
    // bool is_robot_running = packet->robot_status & 0x02;
    // bool is_bumper1_on    = packet->robot_status & 0x04;
    // bool is_bumper2_on    = packet->robot_status & 0x08;
    // bool is_on_charge     = packet->robot_status & 0x40;
    // bool is_estop_enabled = packet->robot_status & 0x80;
    // bool is_robot_running = packet->robot_status & 0x40;
    // bool is_bumper1_on    = packet->robot_status & 0x20;
    // bool is_bumper2_on    = packet->robot_status & 0x10;
    // bool is_on_charge     = packet->robot_status & 0x02;

    // std::cout << "bat_percent: " << (int)packet->bat_percent << std::endl;
    // std::cout << "robot_status: " << (int)packet->robot_status << std::endl;
    // std::cout << "  is_estop_enabled: " << is_estop_enabled << std::endl;
    // std::cout << "  is_robot_running: " << is_robot_running << std::endl;
    // std::cout << "  is_bumper1_on: " << is_bumper1_on << std::endl;
    // std::cout << "  is_bumper2_on: " << is_bumper2_on << std::endl;
    // std::cout << "  is_on_charge: " << is_on_charge << std::endl;
    // std::cout << "linear_speed: " << (int)packet->linear_speed << std::endl;
    // std::cout << "angular_speed: " << (int)packet->angular_speed << std::endl;

    mf_msgs::msg::MD200Status msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = "md200";
    msg.bat_percent = packet->bat_percent;

    serial_pub->publish(msg);
    return;
  }

  Serial_MD200& set_print_raw(bool val){
    enable_print_raw = val;
    return *this;
  }

private:
  rclcpp::Subscription<mf_msgs::msg::MD200Control>::SharedPtr serial_sub;
  rclcpp::Publisher<mf_msgs::msg::MD200Status>::SharedPtr serial_pub;
  bool enable_print_raw = false;
};

class Serial_Lift1400: public MfSerialBase{
public:
  Serial_Lift1400(std::string node_name)
    :MfSerialBase(node_name)
  {
    serial_set_sub = this->create_subscription<mf_msgs::msg::LiftControl>("lift/in", 1, std::bind(&Serial_Lift1400::topic_set_callback, this, std::placeholders::_1));
    serial_pub = this->create_publisher<mf_msgs::msg::LiftStatus>("lift/out", 1);

    // 10: PID, 11: data : start monitoring
    std::vector<uint8_t> init_data;
    init_data.push_back(183);
    init_data.push_back(184);
    init_data.push_back(1);
    init_data.push_back(10);//PID
    init_data.push_back(1);//data
    init_data.push_back(11);//data
    uint8_t check_sum = get_check_sum_md200(init_data.size(), init_data);
    init_data.push_back(check_sum);
    set_write_buffer(init_data);
  }

  ~Serial_Lift1400(){
    serial_set_sub = nullptr;
    std::vector<uint8_t> finish_data;
    // estop
    finish_data.push_back(183);
    finish_data.push_back(184);
    finish_data.push_back(1);
    finish_data.push_back(10);//PID
    finish_data.push_back(1);//data length
    finish_data.push_back(4);//data motor stop
    uint8_t check_sum = get_check_sum_md200(finish_data.size(), finish_data);
    finish_data.push_back(check_sum);

    for (size_t i = 0 ; i < 10; ++i){
      this->_force_write(finish_data);
      usleep(100000);
    }
  }

  void topic_set_callback(const mf_msgs::msg::LiftControl::SharedPtr msg){
    std::vector<uint8_t> data = create_tx_packet_lift(
      msg->cmd_type,            // target_pid
      msg->height_mm,           // height_mm
      msg->speed,               // speed
      msg->accel                // accel
    );
    if (getVerbose()){
      std::cout << "data: " << data.size() << std::endl;
      for (size_t i = 0; i < data.size(); ++i){
        std::cout << "data[" << i << "]: " << (int)data[i] << std::endl;
      }
    }
    set_write_buffer(data);
  }

  uint32_t parse_received_data(const std::vector<uint8_t>& read_buffer_tmp) {
      if (read_buffer_tmp.size() < 16) {  // 최소 길이 검증
          std::cerr << "Not enough data received." << std::endl;
          return 0;
      }

      uint32_t position_raw = read_buffer_tmp[12] |
                              (read_buffer_tmp[13] << 8) |
                              (read_buffer_tmp[14] << 16) |
                              (read_buffer_tmp[15] << 24);

      uint32_t position = static_cast<uint32_t>(position_raw / SCALE_TO_MM);

      return position;
  }


  bool is_rx_msg_valid(const std::deque<uint8_t>& data, int& start_idx, int& end_idx) override {
      // std::cout << "Data size: " << data.size() << std::endl;
      if (data.size() < LIFT_FRAME_LENGTH) {
        if (getVerbose()){
          std::cout << "Data size is less than LIFT_FRAME_LENGTH (" << LIFT_FRAME_LENGTH << ")." << std::endl;
        }
        return false;
      }

      for (size_t st = 0; st < data.size() - LIFT_FRAME_LENGTH; ++st) {
          if (data[st] != RMID_LIFT_TO_PC || data[st + 1] != TMID_LIFT_TO_PC) {
              continue;
          }

          uint8_t length_received_data = data[st + 4];
          size_t total_frame_length = length_received_data + 6;

          if (st + total_frame_length > data.size()) {
              if (getVerbose()){
                std::cout << "Incomplete data frame, skipping." << std::endl;
              }
              continue;
          }

          std::vector<uint8_t> data_tmp(data.begin() + st, data.begin() + st + total_frame_length - 1);
          uint8_t checksum = get_check_sum_md200(data_tmp.size(), data_tmp);
          if (checksum != data[st + total_frame_length - 1]) {
              if (getVerbose()){
                std::cout << "Checksum mismatch. Continuing to next frame." << std::endl;
                std::cout << "checksum: " << (int)checksum << std::endl;
              };
              continue;
          };

          start_idx = static_cast<int>(st);
          end_idx = st + total_frame_length;
          // std::cout << "Valid message found: start_idx=" << start_idx << ", end_idx=" << end_idx << std::endl;
          return true;
      };

      return false;
  }

  void read_callback_on_success(const std::deque<uint8_t>& data) override{
    uint8_t data_array[23];
    std::memcpy(data_array, &data[0], 23);
    std::vector<uint8_t> read_buffer(data_array, data_array + 23);
    uint32_t position = parse_received_data(read_buffer);


    mf_msgs::msg::LiftStatus msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = "lift";
    msg.status_height_mm = position;

    serial_pub->publish(msg);
    return;
  };
private:
  rclcpp::Subscription<mf_msgs::msg::LiftControl>::SharedPtr serial_set_sub;
  rclcpp::Publisher<mf_msgs::msg::LiftStatus>::SharedPtr serial_pub;
};
};
