import rosbag2_py
from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
import rclpy.serialization
from rosidl_runtime_py.utilities import get_message
import os
import shutil
import numpy as np
from sensor_msgs.msg import PointCloud2

# 입력 및 출력 bag 경로
in_bag = '/workspace/data/do_2_active'
out_bag = in_bag + '_modified'

EPOCH_TIME_TO_2024_10_25_12_00_00 = 1727784000000000000

def add_time_offset_to_stamp(stamp, offset_ns):
    """ROS2 메시지의 header.stamp에 시간 오프셋을 더하는 함수."""
    total_ns = stamp.sec * 1_000_000_000 + stamp.nanosec + offset_ns
    stamp.sec = int(total_ns // 1_000_000_000)
    stamp.nanosec = int(total_ns % 1_000_000_000)

def modify_pointcloud_timestamps(msg, offset_ns):
    """PointCloud2 메시지 내 각 포인트의 timestamp를 수정 (float64)."""
    cloud_array = np.frombuffer(msg.data, dtype=np.uint8).copy()

    # timestamp가 있는 필드 찾기 (time 또는 timestamp)
    time_field = next((f for f in msg.fields if f.name in ['time', 'timestamp']), None)

    if time_field:
        offset = time_field.offset  # timestamp 필드의 시작 위치
        point_step = msg.point_step  # 각 포인트의 간격

        # 모든 포인트의 timestamp 수정 (float64 적용)
        for i in range(msg.width):
            point_offset = i * point_step + offset
            original_time = np.frombuffer(cloud_array[point_offset:point_offset + 8], dtype=np.float64)[0]
            # 나노초 오프셋을 초 단위로 변환하여 더함
            new_time = original_time + offset_ns
            # cloud_array[point_offset:point_offset + 8] = np.array([new_time], dtype=np.float64).tobytes()
            cloud_array[point_offset:point_offset + 8] = np.array([new_time], dtype=np.float64).view(np.uint8)


        # 수정된 데이터로 msg.data 업데이트
        msg.data = cloud_array.tobytes()

def register_topic(writer, topic_metadata):
    """주어진 topic_metadata를 writer에 등록."""
    writer.create_topic(topic_metadata)

def main():
    # 기존 출력 bag이 있으면 삭제
    if os.path.exists(out_bag):
        shutil.rmtree(out_bag)

    # 입력 bag 파일 열기
    storage_options = StorageOptions(uri=in_bag, storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # 출력 bag 파일 설정 및 열기
    writer = SequentialWriter()
    writer.open(
        StorageOptions(uri=out_bag, storage_id='sqlite3'),
        ConverterOptions('', '')
    )

    # 토픽 정보 등록
    topics = reader.get_all_topics_and_types()
    topics_to_dtype = {}
    for topic in topics:
        topic_metadata = TopicMetadata(
            name=topic.name,
            type=topic.type,
            serialization_format='cdr'
        )
        register_topic(writer, topic_metadata)
        topics_to_dtype[topic.name] = get_message(topic.type)

    # 각 메시지 읽고 timestamp 및 header.stamp 수정 후 저장
    start_time = None
    sec10_in_nsec = 1e10
    sec10_in_nsec_idx = 1

    while reader.has_next():
        (topic, data, bag_time) = reader.read_next()

        if start_time is None:
            start_time = bag_time
        if bag_time - start_time > sec10_in_nsec * sec10_in_nsec_idx:
            print(f"Processed {sec10_in_nsec_idx}0 seconds")
            sec10_in_nsec_idx += 1

        # 기존 시간에 오프셋을 더함
        new_bag_time = bag_time + EPOCH_TIME_TO_2024_10_25_12_00_00

        # 메시지 타입 가져오기
        msg_type = topics_to_dtype[topic]
        msg = rclpy.serialization.deserialize_message(data, msg_type)

        # header.stamp 수정
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            add_time_offset_to_stamp(msg.header.stamp, EPOCH_TIME_TO_2024_10_25_12_00_00)

        # /livox/lidar 토픽의 PointCloud2 메시지 내 timestamp 수정
        if topic == '/livox/lidar' and isinstance(msg, PointCloud2):
            modify_pointcloud_timestamps(msg, EPOCH_TIME_TO_2024_10_25_12_00_00)

        # 수정된 메시지를 직렬화하여 저장
        serialized_data = rclpy.serialization.serialize_message(msg)
        writer.write(topic, serialized_data, new_bag_time)

    print("Bag file modification complete.")

if __name__ == '__main__':
    main()
