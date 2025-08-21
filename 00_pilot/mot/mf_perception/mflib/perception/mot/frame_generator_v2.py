import numpy as np
import quaternion
import os
import pickle
from mflib.perception.mot import datatypes
from mflib.perception.mot import common

class ErrorCodeGenerator:
  def __init__(self):
    self.error_code = 0
  def get_and_update_error_code(self):
    ret = self.error_code
    self.error_code += 1
    return ret
  
class FrameGeneratorV2:
  def __init__(self):
    error_code_generator = ErrorCodeGenerator()

    self.REASON_INVALID_ARGUMENT \
      = error_code_generator.get_and_update_error_code()
    self.REASON_EMPTY_QUEUE \
      = error_code_generator.get_and_update_error_code()
    
    # sync rgbd
    self.REASON_RGBD_OLD_RGB \
      = error_code_generator.get_and_update_error_code()
    self.REASON_RGBD_OLD_DEPTH \
      = error_code_generator.get_and_update_error_code()
    
    # odom interpolation
    self.REASON_ODOM_QUEUE_LACKS_DATA \
      = error_code_generator.get_and_update_error_code()
    self.REASON_ODOM_INTERPOLATION_NOT_FEASIBLE \
      = error_code_generator.get_and_update_error_code()
    self.REASON_ODOM_OLD_DEPTH \
      = error_code_generator.get_and_update_error_code()
    self.REASON_ODOM_LARGE_TIME_DIFF \
      = error_code_generator.get_and_update_error_code()
    
    # match rgb and detection
    self.REASON_DET2D_NO_MATCHED_RGB \
      = error_code_generator.get_and_update_error_code()

    self.empty_frame_return_reasons = {}
    self.empty_frame_return_reasons[self.REASON_INVALID_ARGUMENT] \
      = 'invalid argument'
    self.empty_frame_return_reasons[self.REASON_EMPTY_QUEUE] \
      = 'empty queue'
    self.empty_frame_return_reasons[self.REASON_RGBD_OLD_RGB] \
      = 'old rgb (sync rgbd)'
    self.empty_frame_return_reasons[self.REASON_RGBD_OLD_DEPTH] \
      = 'old depth (sync rgbd)'
    self.empty_frame_return_reasons[self.REASON_ODOM_QUEUE_LACKS_DATA] \
      = 'len(odom queue) >= 2 is required'
    self.empty_frame_return_reasons[self.REASON_ODOM_INTERPOLATION_NOT_FEASIBLE] \
      = 'odom interpolation not feasible'
    self.empty_frame_return_reasons[self.REASON_ODOM_OLD_DEPTH] \
      = 'old depth detected when interpolating odom' 
    self.empty_frame_return_reasons[self.REASON_ODOM_LARGE_TIME_DIFF] \
      = 'odom time diff is too large'
    self.empty_frame_return_reasons[self.REASON_DET2D_NO_MATCHED_RGB] \
      = 'no matched rgb'
  
  def get_error_string(self, error_code):
    if error_code in self.empty_frame_return_reasons.keys():
      return self.empty_frame_return_reasons[error_code]
    else:
      return 'unknown error'
  
  def sync_rgbd(self, rgb_queue, depth_queue, squeeze=True, **kwargs):
    # return (SUCCESS, frame) or
    # (FAILURE, reason)
    # assume queue is filled with ascending order of timestamp

    # kwargs
    max_allowed_rgbd_timediff_ms = kwargs.get('max_allowed_rgbd_timediff_ms', 5) # ms

    n_rgb_queue = len(rgb_queue)
    n_depth_queue = len(depth_queue)
    if n_rgb_queue == 0 or n_depth_queue == 0:
      return False, self.REASON_EMPTY_QUEUE

    rgb_idx = 0
    depth_idx = 0
    matched_indicies = []
    last_error_code = None
    while rgb_idx < n_rgb_queue and depth_idx < n_depth_queue:
      rgb = rgb_queue[rgb_idx]
      depth = depth_queue[depth_idx]
      timediff_ms = (rgb.ts - depth.ts).toMilliseconds()
      if abs(timediff_ms) < max_allowed_rgbd_timediff_ms:
        matched_indicies.append((rgb_idx, depth_idx))
        rgb_idx += 1
        depth_idx += 1
        continue
      elif timediff_ms < 0: # ts(rgb) < ts(depth)
        rgb_idx += 1
        last_error_code = self.REASON_RGBD_OLD_RGB
        continue
      else:
        depth_idx += 1 # ts(rgb) > ts(depth)
        last_error_code = self.REASON_RGBD_OLD_DEPTH
        continue

    if len(matched_indicies) == 0:
      return False, last_error_code
    
    is_success = True


    synced = []
    for rgb_idx, depth_idx in matched_indicies:
      ret = common.FrameRGBD()
      ret.rgb = rgb_queue[rgb_idx]
      ret.depth = depth_queue[depth_idx]
      synced.append(ret)

    if squeeze:
      rgb_queue.erase_until(synced[-1][0].ts)
      depth_queue.erase_until(synced[-1][1].ts)

    return is_success, synced

  def sync_rgbd_odom(self, rgb_queue, depth_queue, odom_queue, squeeze=True, **kwargs):
    max_allowed_odom_time_diff_ms = kwargs.get('max_allowed_odom_time_diff_ms', 300) # s

    # check length of odom queue
    n_odom_queue = len(odom_queue)
    if n_odom_queue < 2:
      return False, self.REASON_ODOM_QUEUE_LACKS_DATA
    
    # rgb, depth sanity is checked in sync_rgbd
    is_success, matched_rgbd \
      = self.sync_rgbd(rgb_queue, depth_queue, squeeze=False, **kwargs)
    if not is_success:
      error_code = matched_rgbd
      return False, error_code

    odom_idx_prev = 0
    synced_data = []
    last_error_code = None
    is_interpolate_candidate_remained = True
    for synced in matched_rgbd:
      mf_rgb_data = synced.rgb
      mf_depth_data = synced.depth
      if False == is_interpolate_candidate_remained:
        break

      # find two closest odom data
      ts_depth = mf_depth_data.ts
      if odom_queue[odom_idx_prev].ts > ts_depth:
        last_error_code \
          = self.REASON_ODOM_OLD_DEPTH
        continue
      while odom_idx_prev + 1 < n_odom_queue and odom_queue[odom_idx_prev + 1].ts < ts_depth:
        odom_idx_prev += 1
      if odom_idx_prev + 2 >= n_odom_queue: # why + 2? next data w/ idx_to_count 
        # no more odom data to check
        is_interpolate_candidate_remained = False
        last_error_code \
          = self.REASON_ODOM_INTERPOLATION_NOT_FEASIBLE
        continue

      # interpolate
      odom_prev = odom_queue[odom_idx_prev]
      odom_after = odom_queue[odom_idx_prev + 1]

      ts_odom_prev = odom_prev.ts 
      ts_odom_after = odom_after.ts
      
      assert ts_odom_prev <= ts_depth <= ts_odom_after
      if (ts_odom_after - ts_odom_prev).toMilliseconds() > max_allowed_odom_time_diff_ms:
        last_error_code \
          = self.REASON_ODOM_LARGE_TIME_DIFF
        odom_idx_prev += 1
        continue

      pose_prev, orientation_prev = odom_prev.pose, odom_prev.orientation
      pose_after, orientation_after = odom_after.pose, odom_after.orientation

      dt = ts_odom_after - ts_odom_prev
      t_factor = (ts_depth - ts_odom_prev).toSeconds() / dt.toSeconds()
      
      pose_interp = pose_prev + (pose_after - pose_prev) * t_factor
      q_prev    = quaternion.quaternion(*orientation_prev)
      q_after   = quaternion.quaternion(*orientation_after)
      quat = quaternion.slerp_evaluate(q_prev, q_after, t_factor)
      quat_interp = np.array([quat.w, quat.x, quat.y, quat.z]).astype(np.float32) # strange...
      
      # collect elements
      odom = datatypes.MfMotOdometryData(ts_depth.sec, ts_depth.nsec,
                                          pose_interp, quat_interp, None, None)
      # Todo: interpolate velocity with error handling...

      synced_data.append((mf_rgb_data, mf_depth_data, odom))


    if len(synced_data) == 0:
      return False, last_error_code
    
    if squeeze:
      rgb_queue.erase_until(synced_data[-1][0].ts) # not used anymore
      depth_queue.erase_until(synced_data[-1][1].ts) # not used anymore
      odom_queue.erase_before(synced_data[-1][2].ts)
      
      ret_list = []
      for synced_datum in synced_data:
        ret = common.FrameRGBDWithOdom
        ret.rgb = synced_datum[0]
        ret.depth = synced_datum[1]
        ret.odom = synced_datum[2]
        ret_list.append(ret)
      return True, ret_list
    else:
      return True, synced_data

  def sync_rgbd_odom_det2ds(self, \
                            rgb_queue, \
                            depth_queue, \
                            odom_queue, \
                            det2d_queue, 
                            squeeze,
                            **kwargs):
    # return (SUCCESS, frame) or
    # (FAILURE, reason)
    # assume queue is filled with ascending order of timestamp
    squeeze = True

    n_det2d_queue = len(det2d_queue)
    if n_det2d_queue == 0:
      return False, self.REASON_EMPTY_QUEUE
    
    is_success, synced_rgbd_odom \
      = self.sync_rgbd_odom(rgb_queue, 
                            depth_queue, 
                            odom_queue, 
                            squeeze=False,
                            **kwargs)
    if not is_success:
      error_code = synced_rgbd_odom
      return False, error_code
    
    synced_data = []
    last_error_code = None
    idx_det2d = 0
    for idx in range(len(synced_rgbd_odom)):
      ts_rgb = synced_rgbd_odom[idx][0].ts
      # narrowing down time diff of det2d
      while idx_det2d < n_det2d_queue and det2d_queue[idx_det2d].ts < ts_rgb:
        idx_det2d += 1
      
      # check if det2d is too old
      if idx_det2d >= n_det2d_queue: # all det2d is older then synced_rgbd_odom
        last_error_code = self.REASON_DET2D_NO_MATCHED_RGB
        break
      
      if ts_rgb == det2d_queue[idx_det2d].ts:
        synced_data.append((*synced_rgbd_odom[idx], det2d_queue[idx_det2d]))
        continue
    
    if len(synced_data) == 0:
      return False, last_error_code
    else:
      if squeeze:
        rgb_queue.erase_until(synced_data[-1][0].ts)
        depth_queue.erase_until(synced_data[-1][1].ts)
        odom_queue.erase_before(synced_data[-1][2].ts)
        det2d_queue.erase_until(synced_data[-1][3].ts)

      ret_list = []
      for synced_datum in synced_data:
        ret = common.FrameMOT3D()
        ret.rgb = synced_datum[0]
        ret.depth = synced_datum[1]
        ret.odom = synced_datum[2]
        ret.detections = synced_datum[3]
        ret_list.append(ret)
      return True, ret_list
    raise NotImplementedError('This code is not reachable')


class FrameIterator:
  def __init__(self, frames_dir):
    # this class assume all frames are too big to load at once
    if not os.path.exists(frames_dir):
      raise FileNotFoundError(f'{frames_dir} does not exist')
    self.frames_dir = frames_dir
    self.frame_idx = -1 #global counting
    self.frame_idx_in_current_array = -1 # index in current frame array
    self.frame_array_idx = 0 # index of frame array
    self.frames_current_array = []
    
    frame_array_names = os.listdir(frames_dir)
    frame_array_names = [frame_array_name for frame_array_name in frame_array_names if frame_array_name != 'intrinsics.pkl']
    if len(frame_array_names) <= 1:
      raise FileNotFoundError('frames are not correctly created')
    frame_array_names.sort()
    self.frame_array_names = [
      os.path.join(frames_dir, frame_array_name) for frame_array_name in frame_array_names
    ]
    count = 0
    for frame_array_name in self.frame_array_names:
      frame_array = open(frame_array_name, 'rb')
      frame_array = pickle.load(frame_array)
      count += len(frame_array)
    self.count = count
    print('create frame iterator done')

  def __len__(self):
    return self.count
  
  def __iter__(self):
    return self
  
  def __next__(self):
    self.frame_idx += 1
    self.frame_idx_in_current_array += 1

    # end of iteration
    if self.frame_idx >= self.count:
      raise StopIteration
    # beginning of iteration
    if len(self.frames_current_array) == 0:
      frames = open(self.frame_array_names[self.frame_array_idx], 'rb')
      frames = pickle.load(frames)
      self.frames_current_array = frames
      self.frame_array_idx += 1
    if self.frame_idx_in_current_array >= len(self.frames_current_array):
      self.frames_current_array = []
      self.frame_idx_in_current_array = -1
      return self.__next__()
    frame = self.frames_current_array[self.frame_idx_in_current_array]
    ret = self.frame_array_names[self.frame_array_idx - 1], self.frame_idx_in_current_array, frame
    return ret
