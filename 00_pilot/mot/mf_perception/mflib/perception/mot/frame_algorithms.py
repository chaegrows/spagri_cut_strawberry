from mflib.perception.mot.datatypes import MfMotDetection3D
import cv2
import numpy as np
from pykdtree.kdtree import KDTree

# def registerDepthToRGB(frame, **kwargs): # will be fixed in the future...
#   # assume equal or bigger size of depth
#   assert frame.rgb.image.shape[0] <= frame.depth.image.shape[0], 'depth image should be bigger or equal to rgb image'
#   assert frame.rgb.image.shape[1] <= frame.depth.image.shape[1], 'depth image should be bigger or equal to rgb image'

#   depth_unit        = kwargs.get('depth_unit', 'mm') # mm or meter

#   rgb = frame.rgb.image
#   depth = frame.depth.image
#   inf_indicies = (depth > 6000).copy()
#   if depth_unit == 'mm':
#     depth = depth.astype(float)/1000
#   else:
#     raise NotImplementedError(f'depth unit {depth_unit} is not implemented')
#   depth[inf_indicies] = np.nan

#   rgb_K, rgb_D = frame.rgb.K, frame.rgb.D
#   depth_K, depth_D = frame.depth.K, frame.depth.D

#   # register depth to rgb
#   Rt = np.array([[1, 0, 0, 0.015],
#                  [0, 1, 0, 0],
#                  [0, 0, 1, 0],
#                  [0, 0, 0, 1]])
  

#   depth = cv2.rgbd.registerDepth(depth_K.astype(float), 
#                                   rgb_K.astype(float), 
#                                   depth_D.astype(float), 
#                                   Rt.astype(float), 
#                                   depth.astype(float),
#                                 (depth.shape[1], depth.shape[0]),
#                                  depthDilation=True
#                                 ) #this sometimes destroys the depth image, but let me debug later...
#   depth[np.isnan(depth)] = 0
#   depth[np.isinf(depth)] = 0
#   depth[depth < 0] = 0

#   depth = depth[:rgb.shape[0], :rgb.shape[1]]

#   frame.depth.image = depth
#   return frame

def preprocess_depth(depth, depth_min_mm, depth_max_mm, unit_string='mm'):
  # get max of np.    
  if unit_string != 'mm':
    raise NotImplementedError(f'depth unit {unit_string} is not implemented')
  else:
    assert depth.dtype == np.uint16 # mm unit check
    inf_indicies = depth == np.iinfo(depth.dtype).max
    depth = depth.astype(float) / 1000.
    depth[depth < depth_min_mm/1000.] = np.nan
    depth[depth > depth_max_mm/1000.] = np.nan
    depth[inf_indicies] = np.nan
  return depth

def visualize_depth(depth, cv_gui=False, save_path=''):
  if cv_gui == False and save_path == '':
    return
  depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
  depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
  if cv_gui:
    cv2.imshow('depth', depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  if save_path != '':
    cv2.imwrite(save_path, depth)

def convert_with_tf(det3d, transformation_matrix_4x4, frame_id):
  assert isinstance(det3d, MfMotDetection3D), 'measurement should be MfMotFruitCountingMeasurement'
  # convert xyzwlh
  xyz1 = np.array([*det3d.xyzwlh[:3], 1], np.float32)
  xyz1 = np.dot(transformation_matrix_4x4, xyz1)
  xyz1 = xyz1[:3]
  rot = transformation_matrix_4x4[:3, :3]
  wlh = det3d.xyzwlh[3:]
  wlh = np.abs(np.dot(rot, wlh))
  det3d.xyzwlh = np.concatenate([xyz1, wlh])
  if frame_id != '':
    det3d.frame_id = frame_id
  # may consider orientation? complicated operation. ignoring it now since tomatoes or strawberries are ellipsoids
  return det3d


class Detection3DCreatorKeypoint:
  def __init__(self, **kwargs):
    # x, y, conf = keypoint_data

    self.interest_keypoint_labels = kwargs.get('interest_keypoint_labels', [])
    
    self.pseudo_bbox_line_length_pixel \
      = kwargs.get('pseudo_bbox_line_length_pixel', 10)
    self.max_nan_ratio            = kwargs.get('max_nan_ratio', 0.4)
    self.min_confidence           = kwargs.get('min_confidence', 0.5)
    self.min_conf_apply_rule      = kwargs.get('min_conf_apply_rule', 'all')
    
    self.depth_unit               = kwargs.get('depth_unit', 'mm')
    self.min_dist_from_camera     = kwargs.get('min_dist_from_camera', 1.0)
    self.max_dist_from_camera     = kwargs.get('max_dist_from_camera', 1.0)
    if self.depth_unit != 'mm':
      raise NotImplementedError(f'depth unit {self.depth_unit} is not implemented')
    else:
      self.depth_unit_factor = 1000.0

    self.validity_checkers = [
      Detection3DCreatorKeypoint.is_labels_available,
      Detection3DCreatorKeypoint.is_confident
    ]

  def is_labels_available(self, keypoint_data):
    # keypoint labels are increasing order
    if len(self.interest_keypoint_labels) == 0: return True
    n_keypoints = len(keypoint_data.x_ls)
    label_max = max(self.interest_keypoint_labels)
    return n_keypoints >= label_max 
  
  def is_confident(self, keypoint_data):
    confs = np.array(keypoint_data.conf_ls)
    conditions = confs >= self.min_confidence
    if self.min_conf_apply_rule == 'all':
      return conditions.all()
    elif self.min_conf_apply_rule == 'any':
      return conditions.any()

  def create(self, keypoint_data, depth_np, depth_K, depth_D, frame_id=''):
    # this function returns (is_valid, measurement)
    # the list of xyzwlh is in meter, and according to the camera coordinate system
    # check label and bbox

    # assumption: rgb and depth have identical intrinsics
    for checker in self.validity_checkers:
      if not checker(self, keypoint_data):
        return False, checker.__name__
      
    bbox_length = self.pseudo_bbox_line_length_pixel
    rets = []
    for x, y in zip(keypoint_data.x_ls, keypoint_data.y_ls):
      x_i, y_i = int(x), int(y)
      tl = x_i - bbox_length//2, y_i - bbox_length//2
      if tl[0] < 0 or tl[1] < 0: return False, 'bbox out of range'

      x, y, w, h = tl[0], tl[1], bbox_length, bbox_length
      depth_roi = depth_np[y:y+h, x:x+w]
      n_nan = np.isnan(depth_roi).sum()
      n_total = depth_roi.size
      nan_ratio = 1.0*n_nan/n_total
      if nan_ratio > self.max_nan_ratio:
        return False, f'too many nan in depth roi: {nan_ratio*100:.0f}%'

      bbox_depth_1d = depth_roi.reshape(-1, 1)
      bbox_depth_1d = bbox_depth_1d[~np.isnan(bbox_depth_1d)]
      dominant_depth = float(np.median(bbox_depth_1d))
      if dominant_depth < 0.001:
        return False, 'dominant depth is too small'
      
      u = x + w/2
      v = y + h/2
      x_center_meter = (u - depth_K[0, 2]) * dominant_depth / depth_K[0, 0]
      y_center_meter = (v - depth_K[1, 2]) * dominant_depth / depth_K[1, 1]
      z_center_meter = dominant_depth

      # calculate distance from camera
      dist_from_camera \
        = np.sqrt(x_center_meter**2 + y_center_meter**2 + z_center_meter**2)
      if dist_from_camera > self.max_dist_from_camera:
        return False, 'too far from camera'

      width_meter   = w * dominant_depth / depth_K[0, 0]
      height_meter  = h * dominant_depth / depth_K[1, 1]
      length_meter  = (width_meter + height_meter) / 2
      xyzwlh = [x_center_meter, y_center_meter, z_center_meter, width_meter, length_meter, height_meter]
      xyzwlh = np.array(xyzwlh)

      # create measurement
      sec, nsec = keypoint_data.ts.sec, keypoint_data.ts.nsec
      measurement = MfMotDetection3D(sec, nsec, nan_ratio, xyzwlh, frame_id)

      rets.append(measurement)
      
    return True, rets
  

class Detection3DCreaterBBox:
  def __init__(self, **kwargs):
    self.valid_labels             = kwargs.get('valid_labels', [])
    self.valid_borders            = kwargs.get('valid_borders', (0, 0, 0, 0))
    self.min_bbox_size            = kwargs.get('min_bbox_size', (0, 0))
    self.max_bbox_size            = kwargs.get('max_bbox_size', (-1, -1))
    self.image_size_wh            = kwargs.get('image_size_wh', (640, 480))
    self.max_nan_ratio            = kwargs.get('max_nan_ratio', 0.4)
    self.depth_unit               = kwargs.get('depth_unit', 'mm')
    self.out_mask_tlbr            = kwargs.get('out_mask_tlbr', (0, 0, 0, 0))
    self.max_dist_from_camera     = kwargs.get('max_dist_from_camera', 1.0)
    if self.depth_unit != 'mm':
      raise NotImplementedError(f'depth unit {self.depth_unit} is not implemented')
    else:
      self.depth_unit_factor = 1000.0

    self.validity_checkers = [
      Detection3DCreaterBBox.is_valid_label,
      Detection3DCreaterBBox.is_inside_border,
      Detection3DCreaterBBox.is_valid_bbox_size,
    ]
    out_mask_wh = self.out_mask_tlbr[2] - self.out_mask_tlbr[0], \
      self.out_mask_tlbr[3] - self.out_mask_tlbr[1]
    if out_mask_wh[0] > 5 and out_mask_wh[1] > 5:
      self.validity_checkers.append(Detection3DCreaterBBox.is_out_of_mask)

  def is_out_of_mask(self, detection):
    # if no intersection, return True
    tlbr = detection.tlbr
    x1, y1, x2, y2 = tlbr
    x1b, y1b, x2b, y2b = self.out_mask_tlbr
    # get intersection of two box
    x1i, y1i = max(x1, x1b), max(y1, y1b)
    x2i, y2i = min(x2, x2b), min(y2, y2b)
    if x1i > x2i or y1i > y2i:
      return True
    return False

  
  def is_valid_label(self, detection):
    if len(self.valid_labels) == 0: return True
    return detection.label in self.valid_labels
  
  def is_inside_border(self, detection):
    tlbr = detection.tlbr
    x1, y1, x2, y2 = tlbr
    x1b, y1b, x2b, y2b = self.valid_borders
    x2b = self.image_size_wh[0] + x2b
    y2b = self.image_size_wh[1] + y2b
    if x1 < x1b or y1 < y1b : return False
    if x2 > x2b or y2 > y2b : return False
    return True
  
  def is_valid_bbox_size(self, detection):
    _, _, w, h = detection.tlwh
    min_w, min_h = self.min_bbox_size
    max_w, max_h = self.max_bbox_size
    if w < min_w or h < min_h: return False
    if max_w != -1 and w > max_w: return False
    if max_h != -1 and h > max_h: return False
    return True

  def create(self, detection, depth_np, depth_K, depth_D, frame_id=''):
    # this function returns (is_valid, measurement)
    # the xyzwlh is in meter, and according to the camera coordinate system
    # check label and bbox

    # assumption: rgb and depth have identical intrinsics
    for checker in self.validity_checkers:
      if not checker(self, detection):
        return False, checker.__name__
      
    tlwh = detection.tlwh
    x, y, w, h = tlwh.astype(int)
    depth_roi = depth_np[y:y+h, x:x+w]
    n_nan = np.isnan(depth_roi).sum()
    n_total = depth_roi.size
    nan_ratio = 1.0*n_nan/n_total
    if nan_ratio > self.max_nan_ratio:
      return False, f'too many nan in depth roi: {nan_ratio*100:.0f}%'

    bbox_depth_1d = depth_roi.reshape(-1, 1)
    bbox_depth_1d = bbox_depth_1d[~np.isnan(bbox_depth_1d)]
    dominant_depth = float(np.median(bbox_depth_1d))
    if dominant_depth < 0.001:
      return False, 'dominant depth is too small'
    
    u = x + w/2
    v = y + h/2
    x_center_meter = (u - depth_K[0, 2]) * dominant_depth / depth_K[0, 0]
    y_center_meter = (v - depth_K[1, 2]) * dominant_depth / depth_K[1, 1]
    z_center_meter = dominant_depth

    # calculate distance from camera
    dist_from_camera \
      = np.sqrt(x_center_meter**2 + y_center_meter**2 + z_center_meter**2)
    if dist_from_camera > self.max_dist_from_camera:
      return False, 'too far from camera'

    width_meter   = w * dominant_depth / depth_K[0, 0]
    height_meter  = h * dominant_depth / depth_K[1, 1]
    length_meter  = (width_meter + height_meter) / 2
    xyzwlh = [x_center_meter, y_center_meter, z_center_meter, width_meter, length_meter, height_meter]
    xyzwlh = np.array(xyzwlh)

    # create measurement
    sec, nsec = detection.ts.sec, detection.ts.nsec
    measurement = MfMotDetection3D(sec, nsec, nan_ratio, xyzwlh, frame_id)
    
    return True, measurement
  
class CostMetrics:
  def __init__(self, **kwargs):
    self.params = {}
    self.params['use_cube_iou']               = kwargs.get('use_cube_iou', True)
    self.params['weight_cube_iou']            = kwargs.get('weight_cube_iou', 1)
    self.params['use_cube_dist_l2']           = kwargs.get('use_cube_dist_l2', True)
    self.params['weight_cube_dist_l2']        = kwargs.get('weight_cube_dist_l2', 1)
    self.params['max_allowed_cube_dist_l2']   = kwargs.get('max_allowed_cube_dist_l2', 0.01)
    self.params['use_cube_dist_size_l2']      = kwargs.get('use_cube_dist_size_l2', True)
    self.params['weight_cube_dist_size_l2']   = kwargs.get('weight_cube_dist_size_l2', 1)

  def calc_cost_traj_det(self, traj, det):
    cost = 0
    if self.params['use_cube_iou']:
      iou = self.get_iou_xyzwlh_xyzwlh(traj['xyzwlh'], det.xyzwlh)
      cost += self.params['weight_cube_iou'] * (1 - iou)
    if self.params['use_cube_dist_l2']:
      dist = self.get_dist_l2_xyzwlh_xyzwlh(traj['xyzwlh'][:3], det.xyzwlh[:3])
      c = min(dist/self.params['max_allowed_cube_dist_l2'], 1)
      cost += self.params['weight_cube_dist_l2'] * c
    if self.params['use_cube_dist_size_l2']:
      dist = self.get_dist_size_l2_xyzwlh_xyzwlh(traj['xyzwlh'][:3], det.xyzwlh[:3])
      cost += self.params['weight_cube_dist_size_l2'] * dist
    return cost

  def volume_wlh(self, w, l, h):
    return w * l * h

  def get_iou_xyzwlh_xyzwlh(self, xyzwlh1, xyzwlh2):
    intersection = self.get_intersection_xyzwlh_xyzwlh(xyzwlh1, xyzwlh2)
    if intersection is None:
      return 0.0
    else:
      volume1 = self.volume_wlh(*xyzwlh1[3:])
      volume2 = self.volume_wlh(*xyzwlh2[3:])
      volume_intersection = self.volume_wlh(*intersection[3:])
      iou = volume_intersection / (volume1 + volume2 - volume_intersection)
      return iou

  def get_intersection_xyzwlh_xyzwlh(self, xyzwlh1, xyzwlh2):
    xyz1_min = xyzwlh1[:3] - xyzwlh1[3:]/2
    xyz1_max = xyzwlh1[:3] + xyzwlh1[3:]/2
    xyz2_min = xyzwlh2[:3] - xyzwlh2[3:]/2
    xyz2_max = xyzwlh2[:3] + xyzwlh2[3:]/2

    c1c2 = np.maximum(xyz1_min, xyz2_min), np.minimum(xyz1_max, xyz2_max)    
    
    center = (c1c2[0] + c1c2[1]) / 2
    size = c1c2[1] - c1c2[0]
    if np.any(size < 0.00001):
      return None
    else:
      xyzwlh = np.concatenate([center, size])
      return xyzwlh
  
  def get_dist_l2_xyzwlh_xyzwlh(self, xyzwlh1, xyzwlh2):
    # get distance between two xyzwlhs
    x1, y1, z1 = xyzwlh1[:3]
    x2, y2, z2 = xyzwlh2[:3]
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    return dist
  
  def get_dist_size_l2_xyzwlh_xyzwlh(self, xyzwlh1, xyzwlh2):
    # get distance between two xyzwlhs
    w1, l1, h1 = xyzwlh1[3:]
    w2, l2, h2 = xyzwlh2[3:]
    dw = w1 - w2
    dl = l1 - l2
    dh = h1 - h2
    dist = np.sqrt(dw**2 + dl**2 + dh**2)
    return dist

class Trajectory3DSet:
  LIFE_UNINITIALIZED = -1
  LIFE_ALIVE = 0
  LIFE_CONFIRMED = 1

  def reset_db(self):
    self.data = np.zeros(self.N_MAX_TRAJECTORY, self.data_dtype)
    self.n_trajectory = 0
    self.det3ds_in_trajectory = []
    self.confirmed_trajectory_count = 0
    self.data['state_life'] = Trajectory3DSet.LIFE_UNINITIALIZED

  def handle_no_updated_trajectory(self, traj_indicies_to_update):
    live_trajectories = self.get_live_trajectories()

    for traj in live_trajectories:
      traj_index = traj['trajectory_index']
      if traj_index not in traj_indicies_to_update:
        self.data[traj_index]['no_update_count'] += 1
      else:
        self.data[traj_index]['no_update_count'] = 0

    for traj in live_trajectories:
      traj_index = traj['trajectory_index']
      if traj['no_update_count'] > self.params['pruning_no_update_count_thresh']:
        self.data[traj_index]['state_life'] = Trajectory3DSet.LIFE_UNINITIALIZED
    

  def __init__(self, **kwargs):
    self.confirmed_trajectory_count = 0
    self.data_dtype = [
      ('is_updated_this_frame', bool),

      ('xyzwlh', np.float32, 6),
      ('score', np.float32),
      ('score_2d', np.float32),
      ('state_life', np.int32),
      ('det3d_count', np.int32),

      ('trajectory_index', np.int32),
      ('trajectory_id_if_confirmed', np.int32),
      ('no_update_count', np.int32),
    ]
    self.N_MAX_TRAJECTORY = kwargs.get('n_max_trajectory', 10000)
    self.data = np.zeros(self.N_MAX_TRAJECTORY, self.data_dtype)
    self.data['state_life'] = Trajectory3DSet.LIFE_UNINITIALIZED
    self.n_trajectory = 0
    self.det3ds_in_trajectory = [] # trajectory index -> det3ds

    self.params = {}
    self.params['verbose']                        = kwargs.get('verbose', True)
    self.params['pose_update_w']                  = kwargs.get('pose_update_w', 0.5)
    self.params['volume_update_w']                = kwargs.get('volume_update_w', 0.8)
    
    self.params['det2d_update_factor']            = kwargs.get('det2d_update_factor', 0.5)
    self.params['det2d_max_score']                = kwargs.get('det2d_max_score', 0.7)
    self.params['n_required_association']         = kwargs.get('n_required_association', 10)
    self.params['thresh_confirmation']            = kwargs.get('thresh_confirmation', 0.7) 
    self.params['pruning_no_update_count_thresh'] = kwargs.get('pruning_no_update_count', 5)
    
    # score explanation
    # 1 to be ideal (target yolo score, target n_association)
    # 0 to be bad (yolo score 0, n_association 0)

  def get_confirmed_traj_det3ds_label_list(self):
    traj_indicies = self.get_trajectories_by_state(
      Trajectory3DSet.LIFE_CONFIRMED, indicies=True
    )
    rets = []
    for traj_index in traj_indicies:
      det3ds = self.det3ds_in_trajectory[traj_index]
      label = self.get_label(traj_index)
      rets.append((self.data[traj_index], det3ds, label))
    return rets

  def update_with_det3d(self, traj_indicies, det3ds: MfMotDetection3D):
    indicies_double_updated = [] # future work....
    self.data['is_updated_this_frame'] = False
    for traj_index, det3d in zip(traj_indicies, det3ds): # prone to double update in one frame
      traj = self.data[traj_index]
      self.det3ds_in_trajectory[traj_index].append(det3d)
      traj['det3d_count'] += 1

      if traj['is_updated_this_frame']:
        indicies_double_updated.append(traj_index)
      traj['is_updated_this_frame'] = True

      # update xyzwlh
      traj['xyzwlh'][:3] = (1 - self.params['pose_update_w'])*traj['xyzwlh'][:3] + \
                           self.params['pose_update_w']*det3d.xyzwlh[:3]
      traj['xyzwlh'][3:] = (1 - self.params['volume_update_w'])*traj['xyzwlh'][3:] + \
                           self.params['volume_update_w']*det3d.xyzwlh[3:]
      # 2d det score update
      det2d_score = min(det3d.det2d.score, self.params['det2d_max_score'])
      traj['score_2d'] = (1 - self.params['det2d_update_factor']) * (1 - traj['score_2d']) + \
                         self.params['det2d_update_factor'] * det2d_score
      
      # score update
      # tuning guide: check yolo score to see when yolo become confident
      # after that, check the number of association
      score_2d_eval = traj['score_2d'] / self.params['det2d_max_score'] # normalize
      score_n_association = traj['det3d_count'] / self.params['n_required_association']
      score_n_association = min(1.0, score_n_association)
      traj['score'] = score_2d_eval * score_n_association
      
      # lifecycle management
      if traj['state_life'] == Trajectory3DSet.LIFE_ALIVE:
        if traj['score'] >= self.params['thresh_confirmation']:
          traj['state_life'] = Trajectory3DSet.LIFE_CONFIRMED
          traj['trajectory_id_if_confirmed'] = self.confirmed_trajectory_count
          self.confirmed_trajectory_count += 1
    if self.params['verbose']:
      if len(indicies_double_updated) > 0:
        print('double updated trajectories:', indicies_double_updated)


  def query_by_distance(self, det3ds_xyz, radius, n_max_neighbor):
    if self.n_trajectory == 0:
      return []
    tree_xyz = KDTree(self.data['xyzwlh'][:self.n_trajectory, :3])
    _, indicies = tree_xyz.query(det3ds_xyz, distance_upper_bound=radius, k=n_max_neighbor)
    # indicies: (n_det3d, n_max_neighbor)

    rets = []
    for i in range(len(indicies)):
      valid_indicies = indicies[i][indicies[i] != self.n_trajectory]
      rets.append(valid_indicies)
    return rets
  
  def get_trajectory_by_index(self, index):
    return self.data[index]

  def add_trajectory(self, det3d: MfMotDetection3D):
    if self.n_trajectory >= self.N_MAX_TRAJECTORY:
      raise NotImplementedError('too many trajectories. need to implement squeeze or manage multiple trees')
    
    traj = self.data[self.n_trajectory]
    traj['is_updated_this_frame'] = True
    traj['xyzwlh'] = det3d.xyzwlh
    traj['score'] = 0
    traj['score_2d'] = min(det3d.det2d.score, self.params['det2d_max_score'])
    traj['state_life'] = Trajectory3DSet.LIFE_ALIVE
    traj['det3d_count'] = 1
    traj['trajectory_index'] = self.n_trajectory
    traj['trajectory_id_if_confirmed'] = -1
    self.det3ds_in_trajectory.append([det3d])
    self.n_trajectory += 1

  def get_indicies_by_min_score(self, min_score):
    indicies = self.data['score'] >= min_score
    return indicies

  def get_label(self, index):
    det3ds = self.det3ds_in_trajectory[index]
    labels = [det3d.det2d.label for det3d in det3ds]
    labels = np.array(labels).astype(int)
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count_idx = np.argmax(counts) # prone to confusing if there are two labels with same count
    return unique_labels[max_count_idx]

  def get_labels(self, boolean_indicies):
    traj_indicies = np.where(boolean_indicies)[0]
    labels = []
    for index in traj_indicies:
      labels.append(self.get_label(index))
    return labels
  
  def get_last_ts(self, index):
    det3ds = self.det3ds_in_trajectory[index]
    return det3ds[-1].ts

  def get_trajectories_by_state(self, target_state_life=LIFE_ALIVE, indicies=False, min_count_thresh=0):
    assert target_state_life in [Trajectory3DSet.LIFE_ALIVE, Trajectory3DSet.LIFE_CONFIRMED], 'target_state_life should be LIFE_ALIVE or LIFE_CONFIRMED'
    alive_indices = self.data['state_life'] == target_state_life
    if indicies:
      indicies_list = np.where(alive_indices)[0]
      return indicies_list
    trajectories = self.data[alive_indices]
    # consider count
    if min_count_thresh > 0:
      alive_indices = np.logical_and(alive_indices, self.data['det3d_count'] >= min_count_thresh)
      trajectories = self.data[alive_indices]
    det3ds = [self.det3ds_in_trajectory[index] for index in range(self.n_trajectory) if alive_indices[index]]
    return trajectories

  def get_live_trajectories(self, indicies=False):
    alive_indices = self.data['state_life'] == Trajectory3DSet.LIFE_ALIVE
    confirmed_indicies = self.data['state_life'] == Trajectory3DSet.LIFE_CONFIRMED
    target_indicies = np.logical_or(alive_indices, confirmed_indicies)
    if indicies:
      indicies_list = np.where(target_indicies)[0]
      return indicies_list
    trajectories = self.data[target_indicies]
    # det3ds = [self.det3ds_in_trajectory[index] for index in range(self.n_trajectory) if target_indicies[index]]
    return trajectories
    
  
  def get_this_frame_updated_trajectory_indices(self):
    return np.where(self.data['is_updated_this_frame'][:self.n_trajectory])[0]

  def get_this_frame_not_updated_trajectory_indices(self):
    return np.where(~self.data['is_updated_this_frame'][:self.n_trajectory])[0]
  
  def get_n_trajectory(self):
    return self.n_trajectory

  def get_trajectory(self, index):
    return self.data[index]

class Trajectory3DManager:
  def save_trajectory_set(self, save_path):
    np.save(save_path, self.trajectory_set.data)

  def reset_db(self):
    self.trajectory_set.reset_db()

  def disable_pruning(self):
    self.params['do_pruning'] = False
  
  def enable_pruning(self):
    self.params['do_pruning'] = True
    
  def __init__(self, **kwargs):
    self.trajectory_set = Trajectory3DSet(**kwargs)

    self.params = {}
    self.params['query_xyz_radius'] \
      = kwargs.get('query_xyz_radius', 0.1) # meter. what is the minimum distance btw two objects?
    self.params['n_max_neighbor'] \
      = kwargs.get('n_max_neighbor', 10) # How many neighbors can be queried that is within the radius?
    self.params['do_associate_cost_thresh'] \
      = kwargs.get('do_associate_cost_thresh', 0.7) # if cost is below this, then associate
    self.params['do_pruning'] \
      = kwargs.get('do_pruning', True)

  def get_trajectory_set(self):
    return self.trajectory_set

  def update_with_det3ds_in_one_frame(self, det3ds, dist_metric):
    # this function must be called in every frame
    # return: n_det3d_associated, n_det3d_not_associated
    if len(det3ds) == 0:
      return 0, 0

    traj_indicies_to_update = []
    det3ds_to_associate = []

    if self.trajectory_set.n_trajectory == 0:
      for det3d in det3ds:
        self.trajectory_set.add_trajectory(det3d)
      return 0, len(det3ds)

    xyzwlhs = np.array([det3d.xyzwlh[:3] for det3d in det3ds], np.float32)
    near_trajectory_indicies_list = self.trajectory_set.query_by_distance(
      xyzwlhs,
      self.params['query_xyz_radius'],
      self.params['n_max_neighbor']
    )

    for det_idx, near_trajectory_indicies in enumerate(near_trajectory_indicies_list):
      det3d = det3ds[det_idx]
      if len(near_trajectory_indicies) == 0:
        self.trajectory_set.add_trajectory(det3d)
        continue

      # calculate cost and select best one
      acceptable_costs = []
      for traj_index in near_trajectory_indicies:
        traj = self.trajectory_set.get_trajectory_by_index(traj_index)
        if traj['state_life'] == Trajectory3DSet.LIFE_UNINITIALIZED:
          continue
        # try to associate
        cost = dist_metric.calc_cost_traj_det(traj, det3d)
        if cost < self.params['do_associate_cost_thresh']:
          acceptable_costs.append((cost, traj_index))

      lowest_cost_traj_index = -1
      if len(acceptable_costs) > 0:
        lowest_cost_traj_index = min(acceptable_costs, key=lambda x: x[0])[1]
      # if no association, create new trajectory
      if lowest_cost_traj_index == -1:
        self.trajectory_set.add_trajectory(det3d)
        continue
      else: # if associated, then update
        traj_indicies_to_update.append(lowest_cost_traj_index)
        det3ds_to_associate.append(det3d)

    # update
    self.trajectory_set.update_with_det3d(traj_indicies_to_update, det3ds_to_associate)

    if self.params['do_pruning']:
      self.trajectory_set.handle_no_updated_trajectory(traj_indicies_to_update)
    
    return len(det3ds_to_associate), len(det3ds) - len(det3ds_to_associate)
  
