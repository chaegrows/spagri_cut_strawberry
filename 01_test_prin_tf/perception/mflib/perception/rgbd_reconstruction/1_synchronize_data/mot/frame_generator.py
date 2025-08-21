import numpy as np
import quaternion
from . import datatypes
from .common import FrameForFruitCounting, FrameRGBD, FrameRGBDWithOdom

class FrameGenerator:
  REASON_FRUIT_COUNTING_INVALID_ARGUMENT     = 0
  REASON_FRUIT_COUNTING_ANY_QUEUE_EMPTY      = 1
  REASON_FRUIT_COUNTING_OLD_DETECTION        = 2
  REASON_FRUIT_COUNTING_WAIT_OTHER_QUEUE     = 3
  REASON_FRUIT_COUNTING_NO_IDENTICAL_TS      = 4
  

  REASON_RGBD_INVALID_ARGUMENT               = 50
  REASON_RGBD_ANY_QUEUE_EMPTY                = 51
  REASON_RGBD_OLD_RGB                        = 52
  REASON_RGBD_ODOM_EARLIER_EXTRAPOLATION_REQUIRED= 53
  REASON_RGBD_ODOM_LATER_EXTRAPOLATION_REQUIRED= 54

  REASON_FRUIT_COUNTING_BROKEN_ASSUMPTION    = 98
  REASON_FRUIT_COUNTING_DEBUG                = 99


  def __init__(self):
    self.generate = {}
    self.generate['fruit counting'] = self.generate_frame_for_fruit_counting
    
    self.empty_frame_return_reasons = {}
    self.empty_frame_return_reasons[FrameGenerator.REASON_FRUIT_COUNTING_INVALID_ARGUMENT] \
      = 'invalid function call argument'
    self.empty_frame_return_reasons[FrameGenerator.REASON_FRUIT_COUNTING_ANY_QUEUE_EMPTY]  \
      = 'Any queue is empty'
    self.empty_frame_return_reasons[FrameGenerator.REASON_FRUIT_COUNTING_OLD_DETECTION]    \
      = 'Old detection'
    self.empty_frame_return_reasons[FrameGenerator.REASON_FRUIT_COUNTING_DEBUG]            \
      = 'Debugging ongoing...'
    self.empty_frame_return_reasons[FrameGenerator.REASON_FRUIT_COUNTING_WAIT_OTHER_QUEUE] \
      = 'Wait for other queue'

    self.empty_frame_return_reasons[FrameGenerator.REASON_RGBD_INVALID_ARGUMENT]           \
      = 'invalid function call argument'
    self.empty_frame_return_reasons[FrameGenerator.REASON_RGBD_ANY_QUEUE_EMPTY]            \
      = 'Any queue is empty'
    self.empty_frame_return_reasons[FrameGenerator.REASON_RGBD_OLD_RGB]                    \
      = 'Old rgb image'
    self.empty_frame_return_reasons[FrameGenerator.REASON_RGBD_ODOM_EARLIER_EXTRAPOLATION_REQUIRED]              \
      = 'Image earlier than odom. Abort since extrapolation required'
    self.empty_frame_return_reasons[FrameGenerator.REASON_RGBD_ODOM_LATER_EXTRAPOLATION_REQUIRED]              \
      = 'Image later than odom. Abort since extrapolation required'

    self.empty_frame_return_reasons[FrameGenerator.REASON_FRUIT_COUNTING_BROKEN_ASSUMPTION]\
      = 'Broken assumption. Did you modify code a lot? check other rejection logics and their orders'
    self.empty_frame_return_reasons[FrameGenerator.REASON_FRUIT_COUNTING_NO_IDENTICAL_TS]      \
      = 'No identical timestamp found between two tss'
    
  
  def get_error_string(self, error_code):
    return self.empty_frame_return_reasons[error_code]
  
  def generate_frame(self, task, *args, **kwargs):
    task = task.lower()
    if task == 'fruit counting':
      return self.generate['fruit counting'](*args, **kwargs)
    
  def generate_frame_rgbd(self, *args, **kwargs):
    # return: (SUCCESS, frame) or (FAILURE, reason)
    if len(args) != 2:
      return False, FrameGenerator.REASON_RGBD_INVALID_ARGUMENT
    
    rgb_queue, depth_queue = args
    verbose = kwargs.get('verbose', False)
    max_timediff_ms = kwargs.get('max_timediff_ms', 5) # ms

    if len(rgb_queue) == 0 or len(depth_queue) == 0:
      if verbose:
        print('len(rgb_queue):', len(rgb_queue))
        print('len(depth_queue):', len(depth_queue))
      return False, FrameGenerator.REASON_RGBD_ANY_QUEUE_EMPTY
    
    # try to sync with rgb
    rgb = rgb_queue[0]
    last_depth_ts = rgb.ts
    depth_idx = 0

    while True:
      depth = depth_queue[depth_idx]
      timediff_ms = (rgb.ts - depth.ts).toMilliseconds()
      if abs(timediff_ms) < max_timediff_ms:
        break
      if last_depth_ts > rgb.ts and depth.ts > rgb.ts:
        rgb_queue.erase_until(rgb.ts)
        return False, FrameGenerator.REASON_RGBD_OLD_RGB
      last_depth_ts = depth.ts
      depth_idx += 1
      if depth_idx >= len(depth_queue):
        return False, FrameGenerator.REASON_RGBD_ANY_QUEUE_EMPTY
      
    # remove elements
    rgb_queue.erase_until(rgb.ts)
    depth_queue.erase_until(depth.ts)

    frame = FrameRGBD(rgb, depth)
    return True, frame
  
  def generate_frame_rgbd_pose(self, *args, **kwargs):
    # return: (SUCCESS, frame) or (FAILURE, reason)
    if len(args) != 3:
      return False, FrameGenerator.REASON_RGBD_INVALID_ARGUMENT
    
    rgb_queue, depth_queue, odom_queue = args
    verbose = kwargs.get('verbose', False)
    max_timediff_ms = kwargs.get('max_timediff_ms', 5) # ms

    if len(rgb_queue) == 0 or len(depth_queue) == 0:
      if verbose:
        print('len(rgb_queue):', len(rgb_queue))
        print('len(depth_queue):', len(depth_queue))
      return False, FrameGenerator.REASON_RGBD_ANY_QUEUE_EMPTY
    
    # try to sync with rgb
    rgb = rgb_queue[0]
    last_depth_ts = rgb.ts
    depth_idx = 0

    while True:
      depth = depth_queue[depth_idx]
      timediff_ms = (rgb.ts - depth.ts).toMilliseconds()
      if abs(timediff_ms) < max_timediff_ms:
        break
      if last_depth_ts > rgb.ts and depth.ts > rgb.ts:
        rgb_queue.erase_until(rgb.ts)
        return False, FrameGenerator.REASON_RGBD_OLD_RGB
      last_depth_ts = depth.ts
      depth_idx += 1
      if depth_idx >= len(depth_queue):
        depth_queue.erase_until(depth.ts)
        return False, FrameGenerator.REASON_RGBD_ANY_QUEUE_EMPTY
    
    # at this line, we can assume rgb and depth are synchronized

    # sync odom
    after_pose_idx = 0
    while after_pose_idx < len(odom_queue) and odom_queue[after_pose_idx].ts < rgb.ts:
      after_pose_idx += 1
    
    if after_pose_idx == 0:
      rgb_queue.erase_until(rgb.ts)
      depth_queue.erase_until(depth.ts)
      return False, FrameGenerator.REASON_RGBD_ODOM_EARLIER_EXTRAPOLATION_REQUIRED
    elif after_pose_idx == len(odom_queue):
      rgb_queue.erase_before(rgb.ts)
      depth_queue.erase_before(depth.ts)
      odom_queue.erase_before(odom_queue[-1].ts)
      return False, FrameGenerator.REASON_RGBD_ODOM_LATER_EXTRAPOLATION_REQUIRED
    
    # print(rgb.ts, after_pose_idx)
    # print(odom_queue[after_pose_idx - 1].ts)
    # print(odom_queue[after_pose_idx].ts)
    # input('')

    pose_after = odom_queue[after_pose_idx].pose
    orientation_after = odom_queue[after_pose_idx].orientation
    pose_prev = odom_queue[after_pose_idx - 1].pose
    orientation_prev = odom_queue[after_pose_idx - 1].orientation

    # interpolate pose and orientation
    dt = odom_queue[after_pose_idx].ts - odom_queue[after_pose_idx - 1].ts
    t_factor = (rgb.ts - odom_queue[after_pose_idx - 1].ts).toSeconds() / dt.toSeconds()
    
    pose_interp = pose_prev + (pose_after - pose_prev) * t_factor
    q_prev    = quaternion.quaternion(*orientation_prev)
    q_after   = quaternion.quaternion(*orientation_after)
    quat = quaternion.slerp_evaluate(q_prev, q_after, t_factor)
    quat_interp = np.array([quat.w, quat.x, quat.y, quat.z]).astype(np.float32) # strange...
    

    # collect elements
    odom      = datatypes.MfMotOdometryData(rgb.ts.sec, rgb.ts.nsec,
                                          pose_interp, quat_interp, None, None)

    # remove elements
    rgb_queue.erase_until(rgb.ts)
    depth_queue.erase_until(depth.ts)
    odom_queue.erase_before(odom_queue[after_pose_idx - 1].ts)
    
    frame = FrameRGBDWithOdom(rgb, depth, odom)
    return True, frame


  def generate_frame_for_fruit_counting(self, *args, **kwargs):
    # return: (SUCCESS, frame) or (FAILURE, reason)

    # args: rgb_queue, depth_queue, detection_queue, odom_queue
    if len(args) != 4:
      return False, FrameGenerator.REASON_FRUIT_COUNTING_INVALID_ARGUMENT
    
    queues = args
    verbose = kwargs.get('verbose', False)

    rgb_queue, depth_queue, detection_queue, odom_queue = queues
    queues_except_detection = [rgb_queue, depth_queue, odom_queue]
    # queues_images = [rgb_queue, depth_queue]
    
    # check if all queues are not empty
    if len(rgb_queue) == 0 or len(depth_queue) == 0 or len(detection_queue) == 0 or len(odom_queue) == 0:
      if verbose:
        print('len(rgb_queue):', len(rgb_queue))
        print('len(depth_queue):', len(depth_queue))
        print('len(detection_queue):', len(detection_queue))
        print('len(odom_queue):', len(odom_queue))
      return False, FrameGenerator.REASON_FRUIT_COUNTING_ANY_QUEUE_EMPTY
    
    # grep one image detection
    image_detection = detection_queue[0]
    image_detection_ts = image_detection.ts

    # time of image detection must be sandwiched between other queues
    min_tss = []
    max_tss = []
    for queue in queues_except_detection:
      min_tss.append(queue[0].ts)
      max_tss.append(queue[-1].ts)
    min_ts = max(min_tss)
    max_ts = min(max_tss)
  
    # print(image_detection_ts) # I hope not to debug these anymore,,, super tricky...
    # print(min_tss)
    # print(image_detection_ts, min_ts, min_tss)

    # old detection. this means computation is not done in time due to slow computation
    if image_detection_ts < min_ts:
      detection_queue.erase_before(min_ts)
      return False, FrameGenerator.REASON_FRUIT_COUNTING_OLD_DETECTION

    # wait for other queue    
    if image_detection_ts > max_ts:
      return False, FrameGenerator.REASON_FRUIT_COUNTING_WAIT_OTHER_QUEUE
    
    # find rgb image with same timestamp
    rgb_idx = -1
    if rgb_queue[0].ts == image_detection_ts:
      rgb_idx = 0
    else:
      rgb_idx = 1
      while rgb_idx < len(rgb_queue) and rgb_queue[rgb_idx].ts < image_detection_ts:
        rgb_idx += 1
      
    if rgb_idx == -1 or rgb_idx >= len(rgb_queue):
      return False, FrameGenerator.REASON_FRUIT_COUNTING_BROKEN_ASSUMPTION
    elif rgb_queue[rgb_idx].ts != image_detection_ts:
      # this happen when yolo detection is received but corresponding rgb image is not received
      # ensuring transmittion is difficult, so we just erase detection
      detection_queue.erase_until(image_detection_ts)
      return False, FrameGenerator.REASON_FRUIT_COUNTING_NO_IDENTICAL_TS

    # find closest element in each image queue
    depth_after_ele_idx = 0
    while depth_after_ele_idx < len(depth_queue)-1 and depth_queue[depth_after_ele_idx].ts <= image_detection_ts:
      depth_after_ele_idx += 1
    if depth_after_ele_idx == 0:
      # we need at least two element to decide the closest one
      # print('no second element')
      return False, FrameGenerator.REASON_FRUIT_COUNTING_WAIT_OTHER_QUEUE 
    candidate1 = depth_queue[depth_after_ele_idx].ts
    candidate2 = depth_queue[depth_after_ele_idx - 1].ts 

    # choose closer one
    abs1 = abs((candidate1 - image_detection_ts).toSeconds())
    abs2 = abs((candidate2 - image_detection_ts).toSeconds())
    if abs1 < abs2: 
      depth_idx = depth_after_ele_idx
    else:
      depth_idx = depth_after_ele_idx - 1 #twtwtwtw

    # find closest (pose, orientation) and interpolate
    after_pose_idx = 0
    while odom_queue[after_pose_idx].ts < image_detection_ts:
      after_pose_idx += 1
    if after_pose_idx == 0:
      return False, FrameGenerator.REASON_FRUIT_COUNTING_WAIT_OTHER_QUEUE
    
    pose_after = odom_queue[after_pose_idx].pose
    orientation_after = odom_queue[after_pose_idx].orientation
    pose_prev = odom_queue[after_pose_idx - 1].pose
    orientation_prev = odom_queue[after_pose_idx - 1].orientation

    # interpolate pose and orientation
    dt = odom_queue[after_pose_idx].ts - odom_queue[after_pose_idx - 1].ts
    t_factor = (image_detection_ts - odom_queue[after_pose_idx - 1].ts).toSeconds() / dt.toSeconds()
    
    pose_interp = pose_prev + (pose_after - pose_prev) * t_factor
    q_prev    = quaternion.quaternion(*orientation_prev)
    q_after   = quaternion.quaternion(*orientation_after)
    quat = quaternion.slerp_evaluate(q_prev, q_after, t_factor)
    quat_interp = np.array([quat.w, quat.x, quat.y, quat.z]).astype(np.float32)

    # collect elements
    rgb       = rgb_queue[rgb_idx]
    depth     = depth_queue[depth_idx]
    odom      = datatypes.MfMotOdometryData(image_detection_ts.sec, image_detection_ts.nsec, 
                                          pose_interp, quat_interp, None, None) #interpolation not working... debug later...
    
    # remove elements
    rgb_queue.erase_before(rgb.ts)
    depth_queue.erase_before(depth.ts)
    detection_queue.erase_until(image_detection_ts)
    odom_queue.erase_before(odom_queue[after_pose_idx - 1].ts)

    # print("succeed generating frame!")
    frame = FrameForFruitCounting(rgb, depth, image_detection, odom)
    return True, frame