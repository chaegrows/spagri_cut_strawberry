import numpy as np
from mflib.perception.mot.common import Timestamp as cTimestamp
import cv2

## definition of data classes

class MfMotImageData:
  def __init__(self, sec, nsec, image, K, D):
    self.ts = cTimestamp(sec, nsec)

    self.image = image
    self.K = K
    self.D = D

  def __repr__(self):
    with_k_str = 'with K' if self.K is not None else ''
    with_d_str = 'with D' if self.D is not None else ''
    return f'MfMotImageData({self.ts}, {self.image.shape}, {with_k_str}, {with_d_str})'

class MfMotOdometryData:
  def __init__(self, sec, nsec, pose, orientation, vel, angvel):
    self.ts = cTimestamp(sec, nsec)

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

class MfMotImageKeypointData:
  def __init__(self, sec, nsec, x_ls, y_ls, conf_ls):
    self.ts = cTimestamp(sec, nsec)
    self.x_ls = x_ls
    self.y_ls = y_ls
    self.conf_ls = conf_ls

  def __repr__(self):
    x_str = f'x={self.x_ls}'
    y_str = f'y={self.y_ls}'
    conf_str = f'conf={self.conf_ls}'
    return f'MfMotImageKeypointData({self.ts}, {x_str}, {y_str}, {conf_str})'

class KeypointMetadata:
  def __init__(self, x, y, conf, label):
    self.x = x
    self.y = y
    self.conf = conf
    self.label = label

class MfMotImageBBoxData:
  def __init__(self, sec, nsec, tlwh, score, label):
    self.ts = cTimestamp(sec, nsec)

    self._tlwh = np.array(tlwh, np.float32)
    self.score = score
    self.label = label

  def __repr__(self):
    tlwh_ele_strs = [f'{ele:.2f}' for ele in self.tlwh]
    tlwh_str = f'tlwh=[{", ".join(tlwh_ele_strs)}]'
    score_str = f'score={self.score:.2f}'
    label_str = f'label={self.label}'
    return f'MfMotImageBBoxData({self.ts}, {tlwh_str}, {score_str}, {label_str})'

  def draw_on_image(self, cv_image, color=(0,0,255), thickness=2, text=''):
    tlbr = self.tlbr_i
    cv_image = cv2.rectangle(cv_image, tuple(tlbr[:2]), tuple(tlbr[2:]), color, thickness)
    if text != '':
      cv2.putText(cv_image, text, tuple(tlbr[:2]), cv2.FONT_HERSHEY_DUPLEX, 1, color)
    return cv_image

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

class MfMotImageBBoxDataArrayData:
  def __init__(self, sec, nsec, image_bbox_array):
    assert isinstance(image_bbox_array, list), 'image_bbox_array must be a list'
    self.ts = cTimestamp(sec, nsec)
    self.image_bbox_array = image_bbox_array

  def __repr__(self):
    return f'MfMotImageBBoxDataArrayData({self.ts}, {len(self.image_bbox_array)} bboxs)'

  def __len__(self):
    return len(self.image_bbox_array)

  def __getitem__(self, index):
    return self.image_bbox_array[index]

class MfMotImageKeypointDataArrayData:
  def __init__(self, sec, nsec, image_keypoint_array):
    assert isinstance(image_keypoint_array, list), 'image_keypoint_array must be a list'
    self.ts = cTimestamp(sec, nsec)
    self.image_keypoint_array = image_keypoint_array

  def __repr__(self):
    return f'MfMotImageKeypointDataArrayData({self.ts}, {len(self.image_keypoint_array)} keypoints)'

  def __len__(self):
    return len(self.image_keypoint_array)

  def __getitem__(self, index):
    return self.image_keypoint_array[index]



###############################
## definition of queue classes
class MfMotQueue:
  def __init__(self, queue_size=-1):
    self.queue_size = queue_size
    self.queue = []

  def reset(self):
    self.queue = []

  def __len__(self):
    return len(self.queue)

  def __getitem__(self, index):
    if self.queue_size >= 0 and index >= len(self.queue):
      raise IndexError(f'index {index} out of range')
    return self.queue[index]

  def _add_data(self, data):
    self.queue.append(data)
    if self.queue_size >= 0 and len(self.queue) > self.queue_size:
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
  def __init__(self, queue_size=-1):
    super().__init__(queue_size)

  def add_image(self, sec, nsec, image, K, D):
    data = MfMotImageData(sec, nsec, image, K, D)
    self._add_data(data)


class MfMotOdometryQueue(MfMotQueue):
  def __init__(self, queue_size=-1):
    super().__init__(queue_size)

  def add_odometry(self, sec, nsec, pose, orientation, vel, angvel): # quaternion
    data = MfMotOdometryData(sec, nsec, pose, orientation, vel, angvel)
    self._add_data(data)

class MfMotImageBBoxDataArrayQueue(MfMotQueue):
  def __init__(self, queue_size=-1):
    super().__init__(queue_size)

  def add_image_bbox_array(self, sec, nsec, i_dets):
    data = MfMotImageBBoxDataArrayData(sec, nsec, i_dets)
    self._add_data(data)

class MfMotImageKeypointDataArrayQueue(MfMotQueue):
  def __init__(self, queue_size=-1):
    super().__init__(queue_size)

  def add_image_keypoint(self, sec, nsec, keypoints):
    data = MfMotImageKeypointDataArrayData(sec, nsec, keypoints)
    self._add_data(data)



class MfMotDetection3D:
  det3d_id = 0
  def __init__(self, sec, nsec, nan_ratio, xyzwlh, frame_id):
    self.ts = cTimestamp(sec, nsec)
    self.nan_ratio = nan_ratio
    self.xyzwlh = xyzwlh
    self.frame_id = frame_id
    self.id = MfMotDetection3D.det3d_id
    MfMotDetection3D.det3d_id += 1

  def add_metadata(self, det2d, framearray_file, framearray_index):
    self.det2d = det2d
    self.framearray_file = framearray_file
    self.framearray_index = framearray_index



  def __repr__(self):
    return f'MfMotDetection3D({self.ts}, {self.frame_id})'
