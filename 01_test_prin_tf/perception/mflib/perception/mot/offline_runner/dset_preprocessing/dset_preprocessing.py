# installed libraries
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
import rosbag2_py
from rosbag2_py import SequentialReader
import cv_bridge
from rclpy.serialization import deserialize_message
import cv2
import pickle
import sys

# developed algorithms
import datatypes as m_dt
import frame_generator as m_fg
import params as P

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

def generate_frames(rgb_queue, depth_queue):
  # create frames
  frames = []
  synchronizer = m_fg.FrameGenerator()
  while True: 
    len_rgb = len(rgb_queue)
    len_depth = len(depth_queue)
    if P.frame_generation_verbose:
      print('len_rgb: ', len_rgb, 'len_depth: ', len_depth)
    if len_rgb == 0 or len_depth == 0:
      break # until creation of frame is not possible
    success, frame = synchronizer.generate_frame_rgbd(
                                                  rgb_queue, 
                                                  depth_queue, 
                                                  verbose=P.frame_generation_verbose,
                                                  max_timediff_ms=P.max_timediff_ms)
    if success:
      frames.append(frame)
    else:
      error_code = frame
      if P.frame_generation_verbose:
        print('Failed to generate frame: ', synchronizer.get_error_string(error_code))
  
  # frame generation is done here
  print('n_frame: ', len(frames), 'timegap: ', frames[-1].ts - frames[0].ts)
  if len(frames) == 0:
    print('Frame generation failed')
    print('enable P.frame_generation_verbose to see the reason')
    print('Maybe topic name mismatch ?')
  return frames

def dump_frames_rgbd(bag_file):
  # create output path
  cautious_mkdir(P.frame_save_dir)

  # create queues
  rgb_queue = m_dt.MfMotImageQueue()
  depth_queue = m_dt.MfMotImageQueue()

  # read bag
  storage_options = rosbag2_py.StorageOptions(uri=bag_file, storage_id='sqlite3')
  converter_options = rosbag2_py.ConverterOptions(
      input_serialization_format='cdr',
      output_serialization_format='cdr'
  )
  reader = SequentialReader()
  reader.open(storage_options, converter_options)

  # load camera intrinsics
  rgb_K, rgb_D, depth_K, depth_D = None, None, None, None
  bridge = cv_bridge.CvBridge()
  wh_rgb, wh_depth = None, None
  while reader.has_next():
    (topic, data, bag_time) = reader.read_next()

    if topic == P.rosbag_topic_set['rgb_info_topic']:
      if rgb_K is None and rgb_D is None:
        ros_msg = deserialize_message(data, CameraInfo)
        rgb_K = np.array(ros_msg.k).reshape((3, 3))
        rgb_D = np.array(ros_msg.d)
        wh_rgb = (ros_msg.width, ros_msg.height)
    if topic == P.rosbag_topic_set['depth_info_topic']:
      if depth_K is None and depth_D is None:
        ros_msg = deserialize_message(data, CameraInfo)
        depth_K = np.array(ros_msg.k).reshape((3, 3))
        depth_D = np.array(ros_msg.d)
        wh_depth = (ros_msg.width, ros_msg.height)
      
    if rgb_K is not None and depth_K is not None:
      break
  
  # find new matrix K
  # I assume RGB and depth have the identical intrinsic
  # if not, please align their camera intrinsics
  if wh_rgb[0] != wh_depth[0] or wh_rgb[1] != wh_depth[1]:
    print('RGB and depth have different resolution. Please align their camera intrinsics')
    print('you need to align RGB and depth')
    print('if not aligned, below code will not work appropriately')
    print('you may want to call cv2.rgbd.registerDepth()')
    raise NotImplementedError
  new_K = cv2.getOptimalNewCameraMatrix(rgb_K, rgb_D, wh_rgb, 1, wh_rgb)[0]
  old_K, old_D = rgb_K, rgb_D
  # save intrinsics
  pickle.dump(new_K, open(f'{P.frame_save_dir}/intrinsics.pkl', 'wb'))

  # fill other queues
  last_processing_time = None
  start_time = None
  frame_id = 0
  reader = SequentialReader()
  reader.open(storage_options, converter_options)
  while reader.has_next():
    (topic, data, bag_time) = reader.read_next()
    if last_processing_time is None:
      last_processing_time = bag_time
      start_time = bag_time
    
    # create frames in every fixed duration
    if (bag_time - start_time) < P.START_TIME*1e9:
      last_processing_time = bag_time
      continue
    elif (bag_time - last_processing_time)/1e9 > P.GENRATE_FRAME_PROCESSING_DURATION:
      frames = generate_frames(rgb_queue, depth_queue)  
      pickle.dump(frames, open(f'{P.frame_save_dir}/frames_{frame_id:05d}.pkl', 'wb'))
      frame_id += 1

      last_processing_time = bag_time

    # fill rgb, depth, and odometry queues
    if topic == P.rosbag_topic_set['depth_topic']:
      is_compressed = True if P.rosbag_topic_set['depth_topic'].endswith('compressedDepth') else False
      if is_compressed:
        msg_type = CompressedImage 
      else:
        msg_type = Image
      ros_msg = deserialize_message(data, msg_type)
      sec = ros_msg.header.stamp.sec
      nsec = ros_msg.header.stamp.nanosec

      if is_compressed:
        np_arr = np.frombuffer(ros_msg.data, np.uint8)
        image = cv2.imdecode(np_arr[12:], cv2.IMREAD_UNCHANGED)
        depth_queue.add_image(sec, nsec, image, None, None)
      else:
        image = bridge.imgmsg_to_cv2(ros_msg)
        image = cv2.undistort(image, old_K, old_D, None, new_K)
        depth_queue.add_image(sec, nsec, image, None, None)
    if topic == P.rosbag_topic_set['rgb_topic']:
      msg_type = CompressedImage if P.rosbag_topic_set['rgb_topic'].endswith('compressed') else Image
      ros_msg = deserialize_message(data, msg_type)
      sec = ros_msg.header.stamp.sec
      nsec = ros_msg.header.stamp.nanosec

      func = bridge.compressed_imgmsg_to_cv2 if isinstance(ros_msg, CompressedImage) else bridge.imgmsg_to_cv2
      image = func(ros_msg)
      image = cv2.undistort(image, old_K, old_D, None, new_K)
      rgb_queue.add_image(sec, nsec, image, None, None)
  
  # create frames for the last part
  frames = generate_frames(rgb_queue, depth_queue)  
  pickle.dump(frames, open(f'{P.frame_save_dir}/frames_{frame_id:05d}.pkl', 'wb'))
  print('dump frames done')

def load_frames_and_mask_by_depth(pixel_rgb):
  cautious_mkdir(P.masked_rgb_save_dir)
  os.mkdir(f'{P.masked_rgb_save_dir}/rgb')
  os.mkdir(f'{P.masked_rgb_save_dir}/rgb_masked')
  os.mkdir(f'{P.masked_rgb_save_dir}/synced')
  frame_iterator = m_fg.FrameIterator(P.frame_save_dir)
  for _, _, frame in frame_iterator:
    rgb_cv = frame.rgb.image
    depth_cv = frame.depth.image

    # get y, x of interest depth
    mask = (depth_cv > P.min_depth_mm) & (depth_cv < P.max_depth_mm)
    
    # create masked rgb
    rgb_masked = rgb_cv.copy()
    rgb_masked[~mask] = pixel_rgb

    # show
    stacked = np.hstack((rgb_cv, rgb_masked))
    cv2.imwrite(f'{P.masked_rgb_save_dir}/rgb/{frame.ts}.png', rgb_cv)
    cv2.imwrite(f'{P.masked_rgb_save_dir}/rgb_masked/{frame.ts}.png', rgb_masked)
    cv2.imwrite(f'{P.masked_rgb_save_dir}/synced/{frame.ts}.png', stacked)


if __name__ == '__main__':
  if sys.argv[1] == 'sync':
    dump_frames_rgbd(sys.argv[2])
  elif sys.argv[1] == 'depth_mask':
    rgb = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    load_frames_and_mask_by_depth(rgb)

