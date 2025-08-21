import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, Image

from rosbag2_py import SequentialReader
import rosbag2_py

import cv_bridge
import cv2
import numpy as np

import shutil
import os
import mot
from mot.frame_generator import FrameGenerator
from scipy.spatial.transform import Rotation as R


# to run synchronized data, you must ensure odometry csv exist
# which have lines like: sec.nsec, r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3 
# and its name must be odom_+bag_name.csv
# e.g.: bag: odom_dongtan_detail_validtime.csv / odom csv: odom_dongtan_detail_validtime.csv

bag_file = '/workspace/icra/cam_dir_8hour/icra_height_low'
csv_file = './odometry.csv'
output_root = '../data/step1_out'
SEQ_START = 0
SEQ_END = 20000

min_depth = 0.2 # meter
max_depth = 1.0 # meter
bridge = cv_bridge.CvBridge()

rgb_topic = '/d405/color/image_rect_raw/compressed'
depth_topic = '/d405/depth/image_rect_raw'
topics = [rgb_topic, depth_topic]

K = np.array([[424.659423828125, 0, 415.6354064941406],
              [0, 424.659423828125, 241.01075744628906],
              [0, 0, 1]])
D = np.array([[0.0
               ,0.0
               ,0.0
               ,0.0
               ,0.0]])
image_size_xy = (848, 480)
new_K = cv2.getOptimalNewCameraMatrix(K, D, image_size_xy, 0)[0]

target_image_size = (848, 480)
x_factor = target_image_size[0] / image_size_xy[0]
y_factor = target_image_size[1] / image_size_xy[1]
new_K_resized = new_K.copy()
new_K_resized[0, 0] *= x_factor
new_K_resized[1, 1] *= y_factor
new_K_resized[0, 2] *= x_factor
new_K_resized[1, 2] *= y_factor

def parse_csv(csv_file):
  # list of 'time, (r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3)'
  with open(csv_file, 'r') as f:
    lines = f.readlines()
    lines = [line.strip().split(',') for line in lines]
    lines = [(float(line[0]), np.array([float(x) for x in line[1:13]])) for line in lines]
  return lines

def preprocess_image(cv_img, K, D, new_K):
  rectified = cv2.undistort(cv_img, K, D, None, new_K)
  resized = cv2.resize(rectified, target_image_size)
  return resized

def save_rgb_msg(cv_img, rgb_save_dir, seq):
  path = os.path.join(rgb_save_dir, f'{seq}.jpg')
  cv2.imwrite(path, cv_img)

def save_depth_msg(cv_img, depth_save_dir, seq):
  path = os.path.join(depth_save_dir, f'{seq}.png')
  cv_img_orig = cv_img.copy()
  cv_img[np.isnan(cv_img)] = 0
  cv_img[np.isinf(cv_img)] = 0
  cv2.imwrite(path, cv_img)

def save_synced_rgb_depth(rgb_img, depth_img, synced_save_dir, seq):
  path = os.path.join(synced_save_dir, f'{seq}.png')
  min_depth_mm = min_depth * 1000
  max_depth_mm = max_depth * 1000
  cv_img = depth_img.copy()
  cv_img[np.isnan(cv_img)] = min_depth_mm
  cv_img[np.isinf(cv_img)] = max_depth_mm
  cv_img[cv_img < min_depth_mm] = min_depth_mm
  cv_img[cv_img > max_depth_mm] = max_depth_mm
  cv_img = (cv_img - min_depth_mm) / (max_depth_mm - min_depth_mm) * 255
  cv_img = cv_img.astype(np.uint8)
  cv_img = cv2.applyColorMap(cv_img, cv2.COLORMAP_JET)

  # concat vertically
  cv_img = np.concatenate((rgb_img, cv_img), axis=0)
  cv2.imwrite(path, cv_img)
  


def extract_and_sync(bag_file, rgb_save_dir, depth_save_dir, 
                     synced_save_dir, tf_save_file, poses):
  
  # directories
  shutil.rmtree(rgb_save_dir, ignore_errors=True)
  shutil.rmtree(depth_save_dir, ignore_errors=True)
  shutil.rmtree(synced_save_dir, ignore_errors=True)
  os.makedirs(rgb_save_dir, exist_ok=True)
  os.makedirs(depth_save_dir, exist_ok=True)
  os.makedirs(synced_save_dir, exist_ok=True)

  # open bag
  storage_options = rosbag2_py.StorageOptions(uri=bag_file, storage_id='sqlite3')
  converter_options = rosbag2_py.ConverterOptions(
      input_serialization_format='cdr',
      output_serialization_format='cdr'
  )
  reader = SequentialReader()
  reader.open(storage_options, converter_options)

  base_time = None
  
  if reader.has_next():
    (topic, data, bag_time) = reader.read_next()
    base_time = bag_time #nanosec

  # synchronizer
  synchronizer = mot.frame_generator.FrameGenerator()

  # queue
  max_queue_len = 1000
  rgb_queue = mot.datatypes.MfMotImageQueue(max_queue_len)
  depth_queue = mot.datatypes.MfMotImageQueue(max_queue_len)

  # create odom queue
  odom_queue = mot.datatypes.MfMotOdometryQueue(1000000000) # infinite queue
  for ts, tf in poses:
    sec = int(ts)
    nsec = int((ts - sec) * 1e9)

    tf = tf.reshape(3, 4)
    trans = tf[:3, 3]
    rot_mat = tf[:3, :3]
    r = R.from_matrix(rot_mat)
    quat = r.as_quat()
    odom_queue.add_odometry(sec, nsec, trans, quat, None, None)
  processing_duration = 10 # seconds

  print('start processing bag file...')
  rgb_seq   = 0 
  depth_seq = 0 
  n_rgb     = 0
  n_depth   = 0

  tf_outfile = open(tf_save_file, 'w')
  while reader.has_next():
    (topic, data, bag_time) = reader.read_next()
    if topic not in topics:
      continue

    if topic == rgb_topic:
      n_rgb += 1
      # deserialize
      ros_msg = deserialize_message(data, CompressedImage)
      sec = ros_msg.header.stamp.sec
      nsec = ros_msg.header.stamp.nanosec

      # add to mot queue
      image = bridge.compressed_imgmsg_to_cv2(ros_msg)
      image = preprocess_image(image, K, D, new_K)
      rgb_queue.add_image(sec, nsec, image, None, None)

    elif topic == depth_topic:
      n_depth += 1
      # deserialize
      ros_msg = deserialize_message(data, Image)
      sec = ros_msg.header.stamp.sec
      nsec = ros_msg.header.stamp.nanosec

      # add to mot queue
      image = bridge.imgmsg_to_cv2(ros_msg)
      image = preprocess_image(image, K, D, new_K)
      depth_queue.add_image(sec, nsec, image, None, None)

      # if len(depth_queue) > 1:
      #   print(depth_queue[-1].ts - depth_queue[-2].ts)


    # halt processing
    if (bag_time - base_time)*1e-9 > processing_duration:
      # synchronize
      while len(rgb_queue) > 0 and len(depth_queue) > 0:
        success, frame = synchronizer.generate_frame_rgbd_pose(rgb_queue, depth_queue, odom_queue, verbose=True)
        if success:
          if rgb_seq < SEQ_START:
            rgb_seq += 1
            depth_seq += 1
            continue
          elif rgb_seq >= SEQ_END:
            break
          seq = rgb_seq
          # image
          save_rgb_msg(frame.rgb.image, rgb_save_dir, f'{seq:06}')
          save_depth_msg(frame.depth.image, depth_save_dir, f'{seq:06}')
          save_synced_rgb_depth(frame.rgb.image, frame.depth.image, synced_save_dir, f'{seq:06}')
          
          # tf
          trans = frame.odom.pose
          quat = frame.odom.orientation
          rot_mat = R.from_quat(quat).as_matrix()
          tf_ls = [rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2], trans[0],
               rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2], trans[1],
                rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2], trans[2],
                0,0,0,1]
          tf_str = str(seq) + ',' + ','.join([str(x) for x in tf_ls])
          tf_outfile.write(tf_str + '\n')

          rgb_seq += 1
          depth_seq += 1
        else:
          # if error_code == FrameGenerator.REASON_RGBD_ODOM_LATER_EXTRAPOLATION_REQUIRED:
          #   break
          # elif error_code == FrameGenerator.REASON_RGBD_ANY_QUEUE_EMPTY:
          #   break
          print('sync failed. ', synchronizer.get_error_string(frame))

      base_time = bag_time
      print(f'processed {(rgb_seq + depth_seq)/2} images. ')
      # input('Press Enter to continue...'
    if rgb_seq >= SEQ_END:
      rgb_queue.reset()
      depth_queue.reset()
      break

    
    
  while len(rgb_queue) > 0 and len(depth_queue) > 0:
    success, frame = synchronizer.generate_frame_rgbd_pose(rgb_queue, depth_queue, odom_queue, verbose=True)
    if success:
      if rgb_seq < SEQ_START:
        rgb_seq += 1
        depth_seq += 1
        continue
      if rgb_seq >= SEQ_END:
        break
      seq = rgb_seq
      save_rgb_msg(frame.rgb.image, rgb_save_dir, f'{seq:06}')
      save_depth_msg(frame.depth.image, depth_save_dir, f'{seq:06}')
      save_synced_rgb_depth(frame.rgb.image, frame.depth.image, synced_save_dir, f'{seq:06}')

      # tf
      trans = frame.odom.pose
      quat = frame.odom.orientation
      rot_mat = R.from_quat(quat).as_matrix()
      tf_ls = [rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2], trans[0],
               rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2], trans[1],
                rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2], trans[2],
                0,0,0,1]
      tf_str = str(seq) + ',' + ','.join([str(x) for x in tf_ls])
      tf_outfile.write(tf_str + '\n')
    else:
      error_code = frame
      if error_code == FrameGenerator.REASON_RGBD_ODOM_LATER_EXTRAPOLATION_REQUIRED:
        break
      elif error_code == FrameGenerator.REASON_RGBD_ANY_QUEUE_EMPTY:
        break
      print('sync failed. ', synchronizer.get_error_string(frame))

  tf_outfile.close()
  print(f'total processed {(rgb_seq + depth_seq)/2} images. ', end='')
  print(f'n_rgb: {n_rgb}, n_depth: {n_depth}')
  print(f'sync success % - rgb: {rgb_seq/n_rgb*100:.2f}%, depth: {depth_seq/n_depth*100:.2f}%')

  del reader

def main():
  # parse poses
  print(f'parsing {csv_file}...')
  poses = parse_csv(csv_file)

  last_dir = bag_file.split('/')[-1]
  rgb_save_dir                =     os.path.join(output_root, f'{last_dir}/rgb')
  depth_save_dir              =     os.path.join(output_root, f'{last_dir}/depth')
  synced_save_dir             =     os.path.join(output_root, f'{last_dir}/synced')
  tf_save_file                =     os.path.join(output_root, f'{last_dir}/tf_synced.txt')
  print(f'processing {bag_file}')
  print('saving to ', rgb_save_dir, '(+depth, synced image, tf) ...')
  extract_and_sync(bag_file, rgb_save_dir, depth_save_dir, synced_save_dir, tf_save_file, poses)

if __name__ == '__main__':
  main()
