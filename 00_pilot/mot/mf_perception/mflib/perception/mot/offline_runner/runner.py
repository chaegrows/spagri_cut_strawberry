import os
import rosbag2_py
from rosbag2_py import SequentialReader
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from rclpy.serialization import deserialize_message
import numpy as np
from scipy.spatial.transform import Rotation 
import cv_bridge
import cv2
import shutil
import pickle

from ultralytics import YOLO

import mflib.brain.mot.common as common
from mflib.brain.mot import frame_generator as m_fg
from mflib.brain.mot import datatypes as m_dt
from mflib.brain.mot import frame_algorithms as m_fa

# from mflib.brain.mot.offline_runner import params_strawberry_mot as P
# from mflib.brain.mot.offline_runner.params_cherrytomato_mot import params as P
from mflib.brain.mot.offline_runner import params_isu as P
from mflib.brain.mot.utils import mot3d_visualizer

import rclpy
# requires concepts to consider memory limitaitons
import time

EXIT_CODE_WRONG_CONFIG = 17

def generate_frames(rgb_queue, depth_queue, odom_queue, detection2d_model):
  # fill detection queue
  detection2d_queue = m_dt.MfMotImageBBoxDataArrayQueue()
  for rgb_type in rgb_queue:
    sec = rgb_type.ts.sec
    nsec = rgb_type.ts.nsec

    image = rgb_type.image
    detections2d = tiled_inference(
      sec, nsec, detection2d_model, image)
    detection2d_queue.add_image_bbox_array(sec, nsec, detections2d)
  
  # create frames
  frames = []
  synchronizer = m_fg.FrameGenerator()
  while True:
    len_rgb = len(rgb_queue)
    len_depth = len(depth_queue)
    len_odom = len(odom_queue)
    len_det = len(detection2d_queue)
    if P.frame_generation_verbose:
      print('len_rgb: ', len_rgb, 'len_depth: ', len_depth, 'len_odom: ', len_odom, 'len_det: ', len_det)
    if len_rgb == 0 or len_depth == 0 or len_odom == 0 or len_det == 0:
      break
    success, frame = synchronizer.generate_frame_rgb_d_detection_odom(
                                                  rgb_queue, 
                                                  depth_queue, 
                                                  detection2d_queue,
                                                  odom_queue,
                                                  verbose=P.frame_generation_verbose)
    if success:
      frames.append(frame)
    else:
      error_code = frame
      # if error_code == m_fg.FrameGenerator.REASON_FRUIT_COUNTING_WAIT_OTHER_QUEUE:
      #   break
      if P.frame_generation_verbose:
        print('Failed to generate frame: ', synchronizer.get_error_string(error_code))
  print('n_frame: ', len(frames))
  if len(frames) == 0:
    print('Frame generation failed')
    print('enable P.frame_generation_verbose to see the reason')
    print('Maybe topic name mismatch ?')
  return frames

def tiled_inference(sec, nsec, model, img_cv, infer_size=(640, 640), batch_size=1):
  im_h, im_w = img_cv.shape[:2]
  infer_h, infer_w = infer_size

  ret = []
  for x in range(0, im_w, infer_w):
    for y in range(0, im_h, infer_h):
      img_tile = img_cv[y:y+infer_h, x:x+infer_w]
      h, w, _ = img_tile.shape

      if h < infer_h or w < infer_w:
        image_to_infer = np.full((infer_h, infer_w, 3), (0, 0, 0), dtype=np.uint8)
        image_to_infer[:h, :w] = img_tile 
      else:
        image_to_infer = img_tile 
      
      detections2d = model.predict(
                          image_to_infer, 
                          imgsz=infer_size, 
                          conf=P.detection2d_score_thresh, 
                          iou=P.detection2d_iou_thresh,
                          verbose=False)
      if detections2d:
        tlbr_set = detections2d[0].boxes.xyxy.cpu().numpy()
        conf_set = detections2d[0].boxes.conf.cpu().numpy()
        label_set = detections2d[0].boxes.cls.cpu().numpy()
        n_det = len(tlbr_set)
        for idx in range(n_det):
          tlbr = tlbr_set[idx]
          conf = conf_set[idx]
          label = label_set[idx]
          tlwh_img = np.array([tlbr[0]+x, tlbr[1]+y, tlbr[2]-tlbr[0], tlbr[3]-tlbr[1]])
          det = m_dt.MfMotImageBBoxData(sec, nsec, tlwh_img, conf, label)
          ret.append(det)
  if P.detection2d_interactive:
    img_to_show = img_cv.copy()
    if ret:
      for det in ret:
        color = P.labels_to_names_and_colors[det.label]['color']
        label_str = P.labels_to_names_and_colors[det.label]['name']
        box_str = f'{label_str}: {det.score:.2f}'
        cv2.rectangle(img_to_show, (det.tlbr_i[0], det.tlbr_i[1]), 
                      (det.tlbr_i[2], det.tlbr_i[3]), color, 2)
        cv2.putText(img_to_show, box_str, (det.tlbr_i[0], det.tlbr_i[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
      cv2.putText(img_to_show, 'No detection', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    img_to_show = cv2.resize(img_to_show, (2560, 1440))
    cv2.imshow('detection', img_to_show)
    cv2.waitKey(1)
  return ret

def update():
  # create detection(3D, reprojected 3D)
  # associate DA and create 3 groups (u. det, u. traj., matched pair)
  # Lifesycle management
  pass

def cautious_mkdir(target_dir):
  if os.path.exists(target_dir):
    if os.path.isdir(target_dir):
      prn = f'path {target_dir} exists. delete and create new one? (y / n)'
      out = input(prn)
      if out.lower() == 'y':
        shutil.rmtree(target_dir)
    elif os.path.isfile(target_dir):
      print(f"path {target_dir} is file. Check your path")
      os._exit(EXIT_CODE_WRONG_CONFIG)
  os.makedirs(target_dir)

def dump_frames():
  # create output path
  cautious_mkdir(P.frame_output_dir)

  # create queues
  rgb_queue = m_dt.MfMotImageQueue()
  depth_queue = m_dt.MfMotImageQueue()
  odom_queue = m_dt.MfMotOdometryQueue()

  # fill odom queue (come along with GLIM)
  # must be world to 'CAM'
  with open(P.cam_tf_csv_file, 'r') as f:
    lines = f.readlines()
    lines = [line.strip().split(',') for line in lines]
    lines = [(float(line[0]), np.array([float(x) for x in line[1:13]])) for line in lines]
  
    for ts, tf in lines:
      sec = int(ts)
      nsec = int((ts - sec) * 1e9)

      tf = tf.reshape(3, 4)
      trans = tf[:3, 3]
      rot_mat = tf[:3, :3]
      r = Rotation.from_matrix(rot_mat)
      quat = r.as_quat()
      odom_queue.add_odometry(sec, nsec, trans, quat, None, None)

  # read bag
  storage_options = rosbag2_py.StorageOptions(uri=P.bag_file, storage_id='sqlite3')
  converter_options = rosbag2_py.ConverterOptions(
      input_serialization_format='cdr',
      output_serialization_format='cdr'
  )
  reader = SequentialReader()
  reader.open(storage_options, converter_options)

  # create frames
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
    os._exit(EXIT_CODE_WRONG_CONFIG)
  new_K = cv2.getOptimalNewCameraMatrix(rgb_K, rgb_D, wh_rgb, 1, wh_rgb)[0]
  old_K, old_D = rgb_K, rgb_D
  # save intrinsics
  pickle.dump(new_K, open(f'{P.frame_output_dir}/intrinsics.pkl', 'wb'))

  # prepare detection model
  # detection2d_model = MFVisionUltralytics(
  #   model_path=P.detection2d_model_path, use_filter=False) 
  detection2d_model = YOLO(P.detection2d_model_path)

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
    elif (bag_time - start_time) < P.START_TIME*1e9:
      last_processing_time = bag_time
      continue
    elif (bag_time - last_processing_time)/1e9 > P.GENRATE_FRAME_PROCESSING_DURATION:
      frames = generate_frames(rgb_queue, depth_queue, odom_queue, detection2d_model)  
      pickle.dump(frames, open(f'{P.frame_output_dir}/frames_{frame_id:05d}.pkl', 'wb'))
      frame_id += 1

      last_processing_time = bag_time

    # if topic == P.rosbag_topic_set['rgb_info_topic']:
    #   if rgb_K is None and rgb_D is None:
    #     ros_msg = deserialize_message(data, CameraInfo)
    #     rgb_K = np.array(ros_msg.k).reshape((3, 3))
    #     rgb_D = np.array(ros_msg.d)
    # if topic == P.rosbag_topic_set['depth_info_topic']:
    #   if depth_K is None and depth_D is None:
    #     ros_msg = deserialize_message(data, CameraInfo)
    #     depth_K = np.array(ros_msg.k).reshape((3, 3))
    #     depth_D = np.array(ros_msg.d)
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
  
  frames = generate_frames(rgb_queue, depth_queue, odom_queue, detection2d_model)  
  pickle.dump(frames, open(f'{P.frame_output_dir}/frames_{frame_id:05d}.pkl', 'wb'))
  print('dump frames done')


def load_frames_and_create_detections3d():
  cautious_mkdir(P.detection3d_output_dir)
  cautious_mkdir(P.detection3d_debug_output_dir)
  
  # load intrinsics
  cam_K = pickle.load(open(f'{P.frame_read_dir}/intrinsics.pkl', 'rb'))

  # detectino3d creator
  iterator = m_fg.FrameIterator(P.frame_read_dir)

  creator = m_fa.Detection3DCreaterBBox(**P.det3d_params)

  detection3d_id = 0
  detections3d_world = []
  n_detections3d = 0
  for frame_order, (path, frame_idx, frame) in enumerate(iterator):
    ts = frame.ts
    rgb = frame.rgb
    depth = frame.depth
    det2ds = frame.detections
    odom = frame.odom

    rgb_cv = rgb.image
    depth_cv = depth.image

    # preprocess depth
    depth_cv_preprocessed = m_fa.preprocess_depth(
      depth_cv, P.depth_min_mm, P.depth_max_mm, P.depth_unit)
    
    # tf
    pose = odom.pose
    quat = odom.orientation
    rot_mat = Rotation.from_quat(quat).as_matrix()

    transform_world_to_cam = np.zeros((4, 4), np.float32)
    transform_world_to_cam[:3, :3] = rot_mat
    transform_world_to_cam[:3, 3] = pose
    transform_world_to_cam[3, 3] = 1
    # transform_world_to_cam = np.linalg.inv(transform_world_to_cam)
    # create 3D detection
    for det2d in det2ds:
      success, det3d = creator.create(det2d, depth_cv_preprocessed, cam_K, None, 'map')
      if not success: 
        if P.detection3d_verbose:
          print('det3d generation failed: ', det3d)
        continue
      det3d.add_metadata(det2d, path, frame_idx)
      det3d = m_fa.convert_with_tf(det3d, transform_world_to_cam, 'map')
      detections3d_world.append(det3d)
      n_detections3d += 1

    if P.detection3d_create_video:
      # collect det2ds in det3ds in this frame
      det2ds_in_det3ds = []
      for det3d in reversed(detections3d_world):
        if det3d.ts != ts:
          break
        det2ds_in_det3ds.append(det3d.det2d)

      rgb_det2d_vis = np.copy(rgb_cv)
      rgb_det2d_vis \
        = mot3d_visualizer.visualize_det2ds_on_cv_image(rgb_det2d_vis, det2ds, P.labels_to_names_and_colors)
      cv2.putText(rgb_det2d_vis, f'2D detections', (10, 40), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255))

      rgb_det3d_vis = np.copy(rgb_cv)
      rgb_det3d_vis \
        = mot3d_visualizer.visualize_det2ds_on_cv_image(rgb_det3d_vis, det2ds_in_det3ds, P.labels_to_names_and_colors)
      cv2.putText(rgb_det3d_vis, f'3D detections (filtered 2D detections)', (10, 40), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255))
      
      depth_vis = np.copy(depth_cv)
      depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_vis, alpha=0.03), cv2.COLORMAP_JET)
      depth_det2d_vis = np.copy(depth_vis)
      depth_det2d_vis \
        = mot3d_visualizer.visualize_det2ds_on_cv_image(depth_det2d_vis, det2ds, P.labels_to_names_and_colors)
      cv2.putText(depth_det2d_vis, f'2D detections', (10, 40), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255))
      
      depth_det3d_vis = np.copy(depth_vis)
      depth_det3d_vis \
        = mot3d_visualizer.visualize_det2ds_on_cv_image(depth_det3d_vis, det2ds_in_det3ds, P.labels_to_names_and_colors)
      cv2.putText(depth_det3d_vis, f'3D detections (filtered 2D detections)', (10, 40), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255))
      
      H, W = rgb_cv.shape[:2]
      im_to_show = np.zeros((H*2, W*2, 3), dtype=np.uint8)

      im_to_show[:H, :W] = rgb_det2d_vis
      im_to_show[H:, :W] = rgb_det3d_vis
      im_to_show[:H, W:] = depth_det2d_vis
      im_to_show[H:, W:] = depth_det3d_vis

      cv2.imwrite(f'{P.detection3d_debug_output_dir}/detection3d_{frame_order:06d}.jpg', im_to_show)
      if P.detection3d_enable_imshow:
        cv2.imshow('detection3d_debug', im_to_show)
        cv2.waitKey(0)

    if len(detections3d_world) > P.N_DETECTION3D_SAVE:
      fname = f'{P.detection3d_output_dir}/detections3d_{detection3d_id:03d}.pkl'
      pickle.dump(detections3d_world[:P.N_DETECTION3D_SAVE], open(fname, 'wb'))
      detection3d_id += 1
      detections3d_world = detections3d_world[P.N_DETECTION3D_SAVE:]
      print('saved', fname)

  fname = f'{P.detection3d_output_dir}/detections3d_{detection3d_id:03d}.pkl'
  pickle.dump(detections3d_world, open(fname, 'wb'))
  print('saved', fname)

  if P.detection3d_create_video:
    os.system(f'ffmpeg -r 30 -i {P.detection3d_debug_output_dir}/detection3d_%06d.jpg -vcodec libx264 {P.detection3d_debug_output_dir}/detection3d_out.mp4')
    os.system(f'rm {P.detection3d_debug_output_dir}/*.jpg')

  print('create 3D detection done. # of detections3d:', n_detections3d)


def load_detections3d_and_create_trajectories():
  cautious_mkdir(P.projected_trajectories_output_dir)

  N_PROCESS = 500
  det3ds_names = os.listdir(P.detection3d_output_dir)
  det3ds_names = [name for name in det3ds_names if name.endswith('.pkl')]
  det3ds_names.sort()

  det3ds = []
  for name in det3ds_names:
    det3ds_tmp = pickle.load(open(f'{P.detection3d_output_dir}/{name}', 'rb'))
    det3ds.extend(det3ds_tmp)

  # divide deections3d by frame  
  det3ds_in_a_frame = []
  det3ds_divided = []
  
  # debug
  frames = []
  frame_getter = common.FrameGetter()
  intrinsics = pickle.load(open(f'{P.frame_read_dir}/intrinsics.pkl', 'rb'))
  # debug done

  last_det3d_ts = None
  for det3d in det3ds:
    if len(det3ds_divided) >= N_PROCESS:
      break
    if last_det3d_ts is None:
      last_det3d_ts = det3d.ts
      frame = frame_getter.get_frame(
        det3d.framearray_file, 
        det3d.framearray_index
        )
      frames.append(frame)
    elif det3d.ts != last_det3d_ts:
      det3ds_divided.append(det3ds_in_a_frame)
      det3ds_in_a_frame = []
      last_det3d_ts = det3d.ts
      frame = frame_getter.get_frame(
        det3d.framearray_file, 
        det3d.framearray_index
        )
      frames.append(frame)
    det3ds_in_a_frame.append(det3d)
    # image
    
  det3ds_divided.append(det3ds_in_a_frame)
  print('length of images: ', len(frames))
  print('det3ds: ', len(det3ds_divided))
  frames_and_det3ds = list(zip(frames, det3ds_divided))
  ##########################################
  # manager, metric
  cost_metric = m_fa.CostMetrics(**P.cost_metric_params)
  trajectory_manager = m_fa.Trajectory3DManager(**P.trajectory_params)
  
  # debug
  visualizer = mot3d_visualizer.TrajectoryVisualizer(P.labels_to_names_and_colors)
  
  for idx, (frame, det3ds_in_a_frame) in enumerate(frames_and_det3ds):
    det3ds_in_a_frame_of_interest = []
    for det3d in det3ds_in_a_frame:
      if det3d.det2d.label in P.det3d_params['valid_labels']:
        det3ds_in_a_frame_of_interest.append(det3d)

    t_update_start = time.time()
    n_merged, n_create = trajectory_manager.update_with_det3ds_in_one_frame(det3ds_in_a_frame_of_interest, cost_metric)
    t_update_end = time.time()
    print(f'time for update MOT data: {t_update_end - t_update_start}')

    # 3D markers
    if P.interactive_trajectory_generation:
      trajectory_set = trajectory_manager.get_trajectory_set()
      valid_trajectories = trajectory_set.get_trajectories_by_state(
        m_fa.Trajectory3DSet.LIFE_CONFIRMED)
      
      if len(valid_trajectories) == 0:
        continue
      labels = trajectory_set.get_labels(valid_trajectories)
      visualizer.publish_trajectories( ### need to be modified
        valid_trajectories, 
        labels, 
        interest_labels=P.det3d_params['valid_labels'],
        do_erase=True)
      visualizer.publish_camera_fov_from_frame(frame, 
                                              P.det3d_params['image_size_wh'],
                                              P.det3d_params['max_dist_from_camera'],
                                              intrinsics)

      # 2D det on image
      im = np.copy(frame.rgb.image)
      im = visualizer.draw_valid_trajectories_on_image(im, frame, intrinsics, valid_trajectories, labels)
      cv2.putText(im, f'projected trajectories', (10, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 3)
      im2 = np.copy(frame.rgb.image)
      im2 = visualizer.draw_det3d_on_image(im2, det3ds_in_a_frame_of_interest)
      cv2.putText(im2, f'3D detections', (10, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 3)
      im_resized = cv2.resize(im, P.vis_trajectory_image_wh)
      cv2.imshow("im", im_resized)
      cv2.imwrite(f'{P.projected_trajectories_output_dir}/projected_trajectories_{idx:06d}.jpg', im)
      # ffmpeg -r 30 -i projected_trajectories_%d.jpg -vcodec libx264  test.mp4
      
      # imgs = np.vstack((im, im2))
      # cv2.imshow("imgs", imgs)

      cv2.waitKey(0)

def main():
  rclpy.init()

  ###### 1. create 3D detection ###########
  # dump_frames()
  # load_frames_and_create_detections3d()
  
  ###### 2. Data association and manage lifecycle ######
  load_detections3d_and_create_trajectories()  

if __name__ == '__main__':
  main()
  # bag read bag