import sys
import os
import shutil
import argparse
import rosbag2_py
from rosbag2_py import SequentialReader
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from rclpy.serialization import deserialize_message
import cv_bridge
import cv2

output_dir = '/tmp/mot_out'
raw_data_dir = {
  'rgb': 'images/rgb', 
  'depth': 'images/depth'
}
# raw /(name of rosbag) / images

# topic names
# how to check? ros2 bag info (rosbag_file)
rgb_topic = '/femto_bolt/color/image_raw/compressed'


def cautious_mkdir(target_dir):
  if os.path.exists(target_dir):
    if os.path.isdir(target_dir):
      prn = f'path {target_dir} exists. delete and create new one? (y / n)'
      out = input(prn)
      if out.lower() == 'y':
        shutil.rmtree(target_dir)
    elif os.path.isfile(target_dir):
      raise(f"path {target_dir} is file. Check your path")
      
  os.makedirs(target_dir)

if __name__=='__main__':
  # argparse 'rosbag_file'
  parser = argparse.ArgumentParser(description='Extract images from a rosbag file')
  parser.add_argument('rosbag_file', type=str, help='Path to the rosbag file')
  args = parser.parse_args()

  rosbag_file = args.rosbag_file
  rosbag_file_stripped = rosbag_file.split('/')[-1]

  raw_root = os.path.join(output_dir, f'raw/{rosbag_file_stripped}')
  if not os.path.exists(raw_root):
    os.makedirs(raw_root)

  rgb_raw_dir = os.path.join(raw_root, raw_data_dir['rgb'])
  cautious_mkdir(rgb_raw_dir)
  depth_raw_dir = os.path.join(raw_root, raw_data_dir['depth'])
  cautious_mkdir(depth_raw_dir)

  # rosbag
  storage_options = rosbag2_py.StorageOptions(uri=rosbag_file, storage_id='sqlite3')
  converter_options = rosbag2_py.ConverterOptions(
      input_serialization_format='cdr',
      output_serialization_format='cdr'
  )
  reader = SequentialReader()
  reader.open(storage_options, converter_options)

  # extract images with logging timestamp
  csv_logs = ['sec,nsec,filename']
  bridge = cv_bridge.CvBridge()
  fileid = 0 
  while reader.has_next():
    (topic, data, bag_time) = reader.read_next()

    if topic == rgb_topic:
      msg_type = CompressedImage 
      ros_msg = deserialize_message(data, msg_type)
      sec = ros_msg.header.stamp.sec
      nsec = ros_msg.header.stamp.nanosec

      image = bridge.compressed_imgmsg_to_cv2(ros_msg)
      im_name = os.path.join(rgb_raw_dir, f'{fileid}.png')
      fileid += 1
      cv2.imwrite(im_name, image)
      csv_logs.append(f'{sec},{nsec},{im_name}')
      print(f'Extracted {im_name}')
  
  # save csv
  csv_file = os.path.join(raw_root, 'ts_rgb.csv')
  with open(csv_file, 'w') as f:
    for line in csv_logs:
      f.write(line + '\n')
  print(f'Saved {csv_file}')