import numpy as np

from . import common

## definition of data classes

class MfMotImageData:
  def __init__(self, sec, nsec, image, K, D):
    self.ts = common.Timestamp(sec, nsec)
    
    self.image = image
    self.K = K
    self.D = D
  
  def __repr__(self):
    with_k_str = 'with K' if self.K is not None else ''
    with_d_str = 'with D' if self.D is not None else ''
    return f'MfMotImageData({self.ts}, {self.image.shape}, {with_k_str}, {with_d_str})'

class MfMotOdometryData:
  def __init__(self, sec, nsec, pose, orientation, vel, angvel):
    self.ts = common.Timestamp(sec, nsec)
    
    self.pose = pose
    self.orientation = orientation
    self.vel = vel
    self.angvel = angvel
  
  def __repr__(self):
    pose_ele_strs = [f'{ele:.2f}' for ele in self.pose]
    pose_str = f'[pose={", ".join(pose_ele_strs)}]'
    orientation_ele_strs = [f'{ele:.2f}' for ele in self.orientation]
    orientation_str = f'[orientation={", ".join(orientation_ele_strs)}]'
    vel_str = f'vel={self.vel:.2f}' if self.vel is not None else ''
    angvel_str = f'angvel={self.angvel:.2f}' if self.angvel is not None else ''
    return f'MfMotOdometryData({self.ts}, {pose_str}, {orientation_str}, {vel_str}, {angvel_str})'
    
class MfMotImageDetectionData:
  def __init__(self, sec, nsec, tlwh, score, label):
    self.ts = common.Timestamp(sec, nsec)
    
    self._tlwh = np.array(tlwh, np.float32)
    self.score = score
    self.label = label
  
  def __repr__(self):
    tlwh_ele_strs = [f'{ele:.2f}' for ele in self.tlwh]
    tlwh_str = f'tlwh=[{", ".join(tlwh_ele_strs)}]'
    score_str = f'score={self.score:.2f}'
    label_str = f'label={self.label}'
    return f'MfMotImageDetectionData({self.ts}, {tlwh_str}, {score_str}, {label_str})'
  
  @property
  def tlwh(self):
    return self._tlwh

  @property
  def tlbr(self):
    return np.concatenate([self.tlwh[:2], self.tlwh[:2] + self.tlwh[2:]])

  @property
  def tlbr_i(self):
    ret = np.concatenate([self.tlwh[:2], self.tlwh[:2] + self.tlwh[2:]])
    return ret.astype(np.int32)

class MfMotImageDetectionArrayData:
  def __init__(self, sec, nsec, image_detection_array):
    assert isinstance(image_detection_array, list), 'image_detection_array must be a list'
    self.ts = common.Timestamp(sec, nsec)
    self.image_detection_array = image_detection_array
  
  def __repr__(self):
    return f'MfMotImageDetectionArrayData({self.ts}, {len(self.image_detection_array)} detections)'
  
  def __len__(self):
    return len(self.image_detection_array)
  
  def __getitem__(self, index):
    return self.image_detection_array[index]
  
class MfMot2DImageTrack:
  def __init__(self, tlwh, track_id, score):
    self._tlwh = tlwh
    self.track_id = track_id
    self.score = score

  @property
  def tlwh(self):
    return self._tlwh

  @property
  def tlbr(self):
    return np.concatenate([self.tlwh[:2], self.tlwh[:2] + self.tlwh[2:]])

  @property
  def tlbr_i(self):
    ret = np.concatenate([self.tlwh[:2], self.tlwh[:2] + self.tlwh[2:]])
    return ret.astype(np.int32)
  
class MfMot2DImageTrackArrayData:
  def __init__(self, sec, nsec, image_track_array):
    assert isinstance(image_track_array, list), 'image_track_array must be a list'
    self.ts = common.Timestamp(sec, nsec)
    self.image_track_array = image_track_array
  
  def __repr__(self):
    return f'MfMot2DImageTrackArrayData({self.ts}, {len(self.image_track_array)} tracks)'
  
  def __len__(self):
    return len(self.image_track_array)
  
  def __getitem__(self, index):
    return self.image_track_array[index]
## definition of queue classes



class MfMotQueue:
  def __init__(self, queue_size):
    self.queue_size = queue_size
    self.queue = []
  
  def reset(self):
    self.queue = []
  
  def __len__(self):
    return len(self.queue)

  def __getitem__(self, index):
    if index >= len(self.queue):
      raise IndexError(f'index {index} out of range')
    return self.queue[index]
  
  def _add_data(self, data):
    self.queue.append(data)
    if len(self.queue) > self.queue_size:
      self.queue.pop(0)
    
  def get_first_timestamp(self):
    if len(self.queue) == 0:
      return None
    return self.queue[0].ts

  def get_last_timestamp(self):
    if len(self.queue) == 0:
      return None
    return self.queue[-1].ts
  
  def erase_before(self, ts):
    while len(self.queue) > 0 and self.queue[0].ts < ts:
      self.queue.pop(0)
  
  def erase_until(self, ts):
    while len(self.queue) > 0 and self.queue[0].ts <= ts:
      self.queue.pop(0)

class MfMotImageQueue(MfMotQueue):
  def __init__(self, queue_size):
    super().__init__(queue_size)

  def add_image(self, sec, nsec, image, K, D):
    data = MfMotImageData(sec, nsec, image, K, D)
    self._add_data(data)

  
class MfMotOdometryQueue(MfMotQueue):
  def __init__(self, queue_size):
    super().__init__(queue_size)
  
  def add_odometry(self, sec, nsec, pose, orientation, vel, angvel): # quaternion
    data = MfMotOdometryData(sec, nsec, pose, orientation, vel, angvel)
    self._add_data(data)

class MfMotImageDetectionArrayQueue(MfMotQueue):
  def __init__(self, queue_size):
    super().__init__(queue_size)
  
  def add_image_detection_array(self, sec, nsec, i_dets):
    data = MfMotImageDetectionArrayData(sec, nsec, i_dets)
    self._add_data(data)


class MfMotFruitCountingMeasurement:
  def __init__(self, sec, nsec, nan_ratio, xyzwlh, rgb_roi, depth_roi):
    self.ts = common.Timestamp(sec, nsec)
    self.nan_ratio = nan_ratio
    self.xyzwlh = xyzwlh
    self.rgb_roi = rgb_roi
    self.depth_roi = depth_roi

  def __repr__(self):
    xyzwlh_ele_strs = [f'{ele:.2f}' for ele in self.xyzwlh]
    xyzwlh_str = f'xyzwlh=[{", ".join(xyzwlh_ele_strs)}]'
    nan_ratio_str = f'nan_ratio={self.nan_ratio:.2f}'
    return f'MfMotFruitCountingMeasurement({self.ts}, {nan_ratio_str}, {xyzwlh_str})'