# todo:
#   update for imu
#   consider cam intrinsics

import argparse
import h5py
import os

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Imu
from sensor_msgs.msg import PointCloud2
import cv_bridge
import cv2  
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d

np.set_printoptions(formatter={'float_kind': lambda x: f"{x:.9f}"})


PATH_ROOT = '/workspace/data/db/'
RGB_FORMAT = 'png'
MAX_STR_LEN = 256




PC2_ROS_TYPE_TO_DTYPE = {
  1: np.int8,
  2: np.uint8,
  3: np.int16,
  4: np.uint16,
  5: np.int32,
  6: np.uint32,
  7: np.float32,
  8: np.float64,
}

RGB_CLASS = 'rgb'
DEPTH_CLASS = 'depth'
LIDAR3D_CLASS = 'lidar3d'
IMU_CLASS = 'imu'

# FN refers to the field name
INDEX_FN_NO_H5 = 'index' # not exposed to h5
DECODE_FUNC_FN_NO_H5 = 'decode_func' # not exposed to h5

CLASS_FN = 'class'
DIRECTORY_FN = 'directory'
FILENAME_SECONDS_FN = 'filename_seconds'
CAM_INTRINSICS_FN = 'camera_intrinsics'

FN_TO_NP_DTYPE = {
  CLASS_FN: np.dtype((f'S{MAX_STR_LEN}')),
  DIRECTORY_FN: np.dtype(f'S{MAX_STR_LEN}'),
  FILENAME_SECONDS_FN: np.dtype([('filename', np.dtype(f'S{MAX_STR_LEN}')), ('second', np.float64)]),
  CAM_INTRINSICS_FN: np.float32,
}

H5_FN_OF_CLASS = {
  RGB_CLASS: {
    CLASS_FN: RGB_CLASS,
    DIRECTORY_FN: '',
    FILENAME_SECONDS_FN: [],
    CAM_INTRINSICS_FN: np.zeros((3, 3), dtype=np.float32),
  },
  DEPTH_CLASS: {
    CLASS_FN: DEPTH_CLASS,
    DIRECTORY_FN: '',
    FILENAME_SECONDS_FN: [],
    CAM_INTRINSICS_FN: np.zeros((3, 3), dtype=np.float32),
  },
  LIDAR3D_CLASS: {
    CLASS_FN: LIDAR3D_CLASS,
    DIRECTORY_FN: '',
    FILENAME_SECONDS_FN: [],  
  }
  # imu: to be updated...
}

class BagDecoder:
  def __init__(self, bag_path, output_root):
    self.bag_path = bag_path
    self.output_root = output_root
    self.topics_to_metadata_dict = {}
    self.cv_bridge = cv_bridge.CvBridge()

  def map_topic_to_decode_func(self):
    bag_reader = BagDecoder.open_bag(self.bag_path)
    topics_info = bag_reader.get_all_topics_and_types()

    for topic in topics_info:
      if topic.type in ['sensor_msgs/msg/CompressedImage', 'sensor_msgs/msg/Image']: # rgb, depth
        if 'color' in topic.name:
          # intrinsic topic name
          prefix = topic.name.split('color')[0]
          intrinsic_topic_name = f'{prefix}color/camera_info'
          self.topics_to_metadata_dict[topic.name] = H5_FN_OF_CLASS[RGB_CLASS].copy()
          self.topics_to_metadata_dict[topic.name][DECODE_FUNC_FN_NO_H5] = self.decode_rgb
          self.topics_to_metadata_dict[topic.name][DIRECTORY_FN] = topic.name.replace('/', '_').lstrip('_')
          
          self.topics_to_metadata_dict[intrinsic_topic_name] = {}
          self.topics_to_metadata_dict[intrinsic_topic_name][CLASS_FN] = CAM_INTRINSICS_FN
          self.topics_to_metadata_dict[intrinsic_topic_name][DECODE_FUNC_FN_NO_H5] = self.decode_intrinsics
        elif 'depth' in topic.name:
          # intrinsic topic name
          prefix = topic.name.split('depth')[0]
          intrinsic_topic_name = f'{prefix}depth/camera_info'
          self.topics_to_metadata_dict[topic.name] = H5_FN_OF_CLASS[DEPTH_CLASS].copy()
          self.topics_to_metadata_dict[topic.name][DECODE_FUNC_FN_NO_H5] = self.decode_depth
          self.topics_to_metadata_dict[topic.name][DIRECTORY_FN] = topic.name.replace('/', '_').lstrip('_')
          self.topics_to_metadata_dict[intrinsic_topic_name] = {}
          self.topics_to_metadata_dict[intrinsic_topic_name][CLASS_FN] = CAM_INTRINSICS_FN
          self.topics_to_metadata_dict[intrinsic_topic_name][DECODE_FUNC_FN_NO_H5] = self.decode_intrinsics
      elif topic.name == '/livox/lidar': #lidar 3d
        self.topics_to_metadata_dict[topic.name] = H5_FN_OF_CLASS[LIDAR3D_CLASS].copy()
        self.topics_to_metadata_dict[topic.name][DECODE_FUNC_FN_NO_H5] = self.decode_lidar
        self.topics_to_metadata_dict[topic.name][DIRECTORY_FN] = topic.name.replace('/', '_').lstrip('_')
      elif topic.type == 'sensor_msgs/msg/Imu': # imu
        self.topics_to_metadata_dict[topic.name] = {}
        self.topics_to_metadata_dict[topic.name][CLASS_FN] = IMU_CLASS
        self.topics_to_metadata_dict[topic.name][DECODE_FUNC_FN_NO_H5] = self.decode_imu
        self.topics_to_metadata_dict[topic.name][DIRECTORY_FN] = topic.name.replace('/', '_').lstrip('_')
    
    for topic_str in self.topics_to_metadata_dict:
      func = self.topics_to_metadata_dict[topic_str]['decode_func']
      print(f'topic: {topic_str}, decoding function: {func.__name__}')
  
  @staticmethod
  def open_bag(bag_path):
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader
  
  def do_decode(self):
    # create dirs
    os.makedirs(self.output_root, exist_ok=True)
    bag_reader = BagDecoder.open_bag(self.bag_path)

    bag_start_time_nsec = None
    next_print_second = 0
    dt_next_print_seconds = 2

    while bag_reader.has_next():
      (topic, data, bag_time_nsec) = bag_reader.read_next()

      # print progress
      if bag_start_time_nsec is None:
        bag_start_time_nsec = bag_time_nsec
        next_print_second += dt_next_print_seconds
        print('bag processing start...')
      else:
        dt_seconds = (bag_time_nsec - bag_start_time_nsec) * 1e-9
        if dt_seconds > next_print_second:
          next_print_second += dt_next_print_seconds 
          print(f'bag processing time: {dt_seconds:.0f} seconds')

      # decode data
      if topic not in self.topics_to_metadata_dict:
        continue
      decode_func = self.topics_to_metadata_dict[topic]['decode_func']
      if decode_func is None:
        raise ValueError(f"decode_func can not be empty. maybe you modified too much?")
      decode_func(topic, data, self.output_root)
    
    print('bag processing done!')
  
    # save metadata as h5
    print('start saving metadata...')
    
    h5file = os.path.join(self.output_root, 'metadata.h5')
    with h5py.File(h5file, "w") as f:
      class_to_fields_grp = f.create_group("class_to_fields")
      
      for class_name, fields in H5_FN_OF_CLASS.items():
        field_names = list(fields.keys())
        class_to_fields_grp.create_dataset(
          class_name, data=np.array(field_names, dtype=f'S{MAX_STR_LEN}')
        )

      for topic, metadata in self.topics_to_metadata_dict.items():
        topic_str = topic.replace('/', '_').lstrip('_')
        class_name = metadata[CLASS_FN]
        if class_name not in H5_FN_OF_CLASS:
          continue

        topic_grp = f.create_group(topic_str)
        for field_name, value in metadata.items():
          if field_name not in H5_FN_OF_CLASS[class_name]:
            continue

          np_dtype = FN_TO_NP_DTYPE[field_name]
          value = np.array(value, dtype=np_dtype) if isinstance(value, list) else value
          topic_grp.create_dataset(field_name, data=value, dtype=np_dtype)

    print('metadata saved!')

  def decode_rgb(self, topic, msg_raw, output_root):
    # filename
    topic_str_replaced = topic.replace('/', '_')
    if topic_str_replaced.startswith('_'):
      topic_str_replaced = topic_str_replaced[1:]
    frame_dir = os.path.join(output_root, topic_str_replaced)
    os.makedirs(frame_dir, exist_ok=True)

    if topic not in self.topics_to_metadata_dict:
      self.topics_to_metadata_dict[topic] = {}
    if INDEX_FN_NO_H5 not in self.topics_to_metadata_dict[topic]:
      self.topics_to_metadata_dict[topic][INDEX_FN_NO_H5] = 0
    filename = f"frame{self.topics_to_metadata_dict[topic][INDEX_FN_NO_H5]:06d}.{RGB_FORMAT}"
    self.topics_to_metadata_dict[topic][INDEX_FN_NO_H5] += 1
    filename = os.path.join(frame_dir, filename)

    # image
    msg_type, convert_func = None, None
    if topic.endswith('compressed'):
      msg_type = CompressedImage
      convert_func = self.cv_bridge.compressed_imgmsg_to_cv2
    else:
      msg_type = Image
      convert_func = self.cv_bridge.imgmsg_to_cv2

    msg = deserialize_message(msg_raw, msg_type)
    sec = msg.header.stamp.sec
    nsec = msg.header.stamp.nanosec
    sec = sec + nsec * 1e-9

    cv_img = convert_func(msg)

    # save image
    cv2.imwrite(filename, cv_img)

    # update for metadata
    self.topics_to_metadata_dict[topic][FILENAME_SECONDS_FN].append((filename, sec))
    return 

  def decode_depth(self, topic, msg_raw, output_root):
    # filename
    topic_str_replaced = topic.replace('/', '_')
    if topic_str_replaced.startswith('_'):
      topic_str_replaced = topic_str_replaced[1:]
    frame_dir = os.path.join(output_root, topic_str_replaced)
    os.makedirs(frame_dir, exist_ok=True)

    if topic not in self.topics_to_metadata_dict:
      self.topics_to_metadata_dict[topic] = {}
    if INDEX_FN_NO_H5 not in self.topics_to_metadata_dict[topic]:
      self.topics_to_metadata_dict[topic][INDEX_FN_NO_H5] = 0
    filename = f"frame{self.topics_to_metadata_dict[topic][INDEX_FN_NO_H5]:06d}.png" # only png is allowed for depth image
    self.topics_to_metadata_dict[topic][INDEX_FN_NO_H5] += 1
    filename = os.path.join(frame_dir, filename)

    # image
    msg = None
    if not topic.endswith('compressedDepth'):
      msg = deserialize_message(msg_raw, Image)
      cv_img = self.cv_bridge.imgmsg_to_cv2(msg)
    else:
      msg = deserialize_message(msg_raw, CompressedImage)
      
      np_arr = np.frombuffer(msg.data, np.uint8)
      cv_img = cv2.imdecode(np_arr[12:], cv2.IMREAD_UNCHANGED)

    sec = msg.header.stamp.sec
    nsec = msg.header.stamp.nanosec
    sec = sec + nsec * 1e-9

    # save image
    cv2.imwrite(filename, cv_img)
    
    # update for metadata
    self.topics_to_metadata_dict[topic][FILENAME_SECONDS_FN].append((filename, sec))
    return 

  def decode_lidar(self, topic, msg_raw, output_root):
    # filename
    topic_str_replaced = topic.replace('/', '_')
    if topic_str_replaced.startswith('_'):
      topic_str_replaced = topic_str_replaced[1:]
    frame_dir = os.path.join(output_root, topic_str_replaced)
    os.makedirs(frame_dir, exist_ok=True)

    if topic not in self.topics_to_metadata_dict:
      self.topics_to_metadata_dict[topic] = {}
    if INDEX_FN_NO_H5 not in self.topics_to_metadata_dict[topic]:
      self.topics_to_metadata_dict[topic][INDEX_FN_NO_H5] = 0
    filename = f"scan{self.topics_to_metadata_dict[topic][INDEX_FN_NO_H5]:06d}" # .pcd is appended later
    self.topics_to_metadata_dict[topic][INDEX_FN_NO_H5] += 1
    filename = os.path.join(frame_dir, filename)

    # decode
    msg = deserialize_message(msg_raw, PointCloud2)
    field_names = [f.name for f in msg.fields] # fields other than x, y, z can not be stored by open3d
    field_map = {f.name: f for f in msg.fields}
    points = list(pc2.read_points(msg, field_names=field_names, skip_nans=True))

    # xyz 
    xyz = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(filename+'.pcd', pcd)

    # others
    meta = {}
    for i, name in enumerate(field_names):
      if name in {"x", "y", "z"}:
        continue
      ros_type = field_map[name].datatype
      np_type = PC2_ROS_TYPE_TO_DTYPE.get(ros_type, np.float32)
      meta[name] = np.array([p[i] for p in points], dtype=np_type)
    np.savez(filename+'.npz', **meta)

    sec = msg.header.stamp.sec
    nsec = msg.header.stamp.nanosec
    sec = sec + nsec * 1e-9

    # update for metadata
    self.topics_to_metadata_dict[topic][FILENAME_SECONDS_FN].append((filename, sec))
    
    return filename, sec

  def decode_imu(self, topic, msg_raw, output_root):
    pass

  def decode_intrinsics(self, topic, msg_raw, output_root): 
    pass

def main():
  parser = argparse.ArgumentParser(description="convert rosbag2 to source")
  parser.add_argument(
    "--bag_path",
    required=True,
    help="Path to the rosbag2 directory (e.g., /workspace/data/db/raw/rosbag2/...)"
  )
  args = parser.parse_args()

  # create dirs
  args.bag_path = args.bag_path.rstrip('/')
  bag_name = args.bag_path.split('/')[-1]
  sensor_output_root = os.path.join(PATH_ROOT, 'source', 'sensors', bag_name)
  decoder = BagDecoder(args.bag_path, sensor_output_root)
  decoder.map_topic_to_decode_func()
  decoder.do_decode()
  

if __name__ == "__main__":
    main()
