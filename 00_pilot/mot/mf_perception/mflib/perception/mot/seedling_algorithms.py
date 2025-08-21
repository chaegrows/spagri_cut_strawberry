import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from mflib.perception.mot.datatypes import MfMotDetection3D

def convert_det3d_with_tf(det3d, transformation_matrix_4x4, frame_id=''):
  assert isinstance(det3d, MfMotDetection3D), 'measurement should be MfMotDetection3D'
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

def preprocess_depth(depth, depth_min, depth_max, unit_string='mm'):
  # get max of np.
  if unit_string != 'mm':
    raise NotImplementedError(f'depth unit {unit_string} is not implemented')
  else:
    assert depth.dtype == np.uint16 # mm unit check
    inf_indicies = depth == np.iinfo(depth.dtype).max
    depth = depth.astype(float) / 1000.
    depth[depth < depth_min] = np.nan
    depth[depth > depth_max] = np.nan
    depth[inf_indicies] = np.nan
  return depth


class TrajectoryArrow2DSet:
  def __init__(self, **kwargs):
    self.confirmed_trajectory_count = 0
    self.data_dtype = [
      ('xyxy', np.float32, 4),
      ('det_count', np.int32),

      ('trajectory_index', np.int32),
      ('trajectory_id_if_confirmed', np.int32),

      ('is_updated_this_frame', bool),
      ('no_update_count', np.int32),
    ]
    self.N_MAX_TRAJECTORY = kwargs.get('n_max_trajectory', 1000)
    self.data = np.zeros(self.N_MAX_TRAJECTORY, self.data_dtype)
    self.n_trajectory = 0
    self.detections_in_trajectories = {k: [] for k in range(self.N_MAX_TRAJECTORY)}

    self.params = {}
    self.params['start_update_w']                  = kwargs.get('pose_update_w', 0.5)
    self.params['end_update_w']                = kwargs.get('volume_update_w', 0.8)

    self.params['n_required_association']         = kwargs.get('n_required_association', 10)
    self.params['pruning_no_update_count_thresh'] = kwargs.get('pruning_no_update_count', 5)

  def reset_db(self):
    self.data = np.zeros(self.N_MAX_TRAJECTORY, self.data_dtype)
    self.n_trajectory = 0
    self.detections_in_trajectories = {k: [] for k in range(self.N_MAX_TRAJECTORY)}
    self.confirmed_trajectory_count = 0

  def get_live_trajectories(self):
    live_indicies = self.data['det_count'] > 0
    return self.data[live_indicies]

  def get_confirmed_trajectories(self):
    confirmed_indicies = self.data['det_count'] > self.params['n_required_association']
    return self.data[confirmed_indicies]

  def add_trajectory(self, arrow2d):
    if self.n_trajectory >= self.N_MAX_TRAJECTORY:
      raise NotImplementedError('too many trajectories')

    self.data[self.n_trajectory]['xyxy'] = arrow2d.xyxy
    self.data[self.n_trajectory]['det_count'] = 1
    self.data[self.n_trajectory]['trajectory_index'] = self.n_trajectory
    self.data[self.n_trajectory]['trajectory_id_if_confirmed'] = -1  # not confirmed yet
    self.data[self.n_trajectory]['is_updated_this_frame'] = True
    self.data[self.n_trajectory]['no_update_count'] = 0
    self.detections_in_trajectories[self.n_trajectory].append(arrow2d)
    self.n_trajectory += 1

  def update_with_arrow2d(self, traj_index, arrow2d):
    traj = self.data[traj_index]

    traj['is_updated_this_frame'] = True
    self.detections_in_trajectories[traj_index].append(arrow2d)
    traj['det_count'] += 1

    # update xyxy
    traj['xyxy'][:2] = (1 - self.params['start_update_w'])*traj['xyxy'][:2] + \
                          self.params['start_update_w']*arrow2d.xyxy[:2]
    traj['xyxy'][2:] = (1 - self.params['end_update_w'])*traj['xyxy'][2:] + \
                          self.params['end_update_w']*arrow2d.xyxy[2:]

    # lifecycle management
    if traj['trajectory_id_if_confirmed'] == -1 and traj['det_count'] >= self.params['n_required_association']:
      traj['trajectory_id_if_confirmed'] = self.confirmed_trajectory_count
      self.confirmed_trajectory_count += 1

  def handle_no_updated_trajectory(self, no_updated_traj_indicies):
    live_trajectories = self.get_live_trajectories()

    for traj in live_trajectories:
      traj_index = traj['trajectory_index']
      if traj_index in no_updated_traj_indicies:
        self.data[traj_index]['no_update_count'] += 1
      else:
        self.data[traj_index]['no_update_count'] = 0

    # prune trajectories that have not been updated for a long time
    target_indicies = self.data['no_update_count'] > self.params['pruning_no_update_count_thresh']
    self.data[target_indicies]['det_count'] = 0  # mark as dead

  def get_tracked_arrows(self):
    confirmed_trajectories = self.get_confirmed_trajectories()
    return confirmed_trajectories

class CostMetrics:
  def __init__(self, **kwargs):
    self.params = {}
    self.params['max_allowed_center_dist'] = kwargs.get('max_allowed_center_dist', 20) # pixel
    self.params['max_allowed_slope_diff'] = kwargs.get('max_allowed_slope_diff', 0.2617993877991494) # slope diff between trajectory and detection
    self.params['center_cost_w'] = kwargs.get('center_cost_weight', 0.5) # weight for center distance cost
    self.params['angle_cost_w'] = kwargs.get('angle_cost_weight', 0.5) # weight for angle difference cost

  def calc_cost_traj_det(self, xyxy_ls1, xyxy_ls2):
    """
    Calculate cost between trajectory and detection boxes.
    :param xyxy_ls1: Trajectory boxes (N, 4) in xyxy format.
    :param xyxy_ls2: Detection boxes (M, 4) in xyxy format.
    :return: Cost matrix (N, M).
    """
    center_cost = self.get_center_dist_costmat(xyxy_ls1, xyxy_ls2)
    angle_cost = self.get_angle_diff_costmat(xyxy_ls1, xyxy_ls2)

    # Combine costs with weights
    cost_matrix = (self.params['center_cost_w'] * center_cost +
                   self.params['angle_cost_w'] * angle_cost)

    return cost_matrix


  def get_center_dist_costmat(self, traj_boxes, det_boxes):
    traj_centers = (traj_boxes[:, :2] + traj_boxes[:, 2:]) / 2.0  # (N, 2)
    det_centers = (det_boxes[:, :2] + det_boxes[:, 2:]) / 2.0      # (M, 2)
    dists = np.linalg.norm(traj_centers[:, None, :] - det_centers[None, :, :], axis=2)
    dists = dists / self.params['max_allowed_center_dist']  # Normalize by max allowed center distance
    dists = np.clip(dists, 0, 1)  # Clip to [0, 1] range
    return dists

  def get_angle_diff_costmat(self, traj_boxes, det_boxes):
    # 각 벡터의 atan2(angle) 계산
    traj_angles = np.arctan2(traj_boxes[:, 3] - traj_boxes[:, 1],  # dy
                             traj_boxes[:, 2] - traj_boxes[:, 0])  # dx  → (N,)

    det_angles = np.arctan2(det_boxes[:, 3] - det_boxes[:, 1],
                            det_boxes[:, 2] - det_boxes[:, 0])     # (M,)

    # 각도 차이 행렬 계산
    angle_diff = np.abs(traj_angles[:, None] - det_angles[None, :])  # (N, M)
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)  # [-pi, pi]로 wrap-around 고려

    # 정규화 및 클리핑
    cost_matrix = angle_diff / self.params['max_allowed_slope_diff']  # Normalize by max allowed slope difference
    cost_matrix = np.clip(cost_matrix, 0, 1)
    return cost_matrix


class TrajectoryArrow2DManager:
  def __init__(self, **kwargs):
    self.trajectory_set = TrajectoryArrow2DSet(**kwargs)
    self.cost_metric = CostMetrics(**kwargs)

    self.params = {}
    self.params['do_associate_cost_thresh'] \
      = kwargs.get('do_associate_cost_thresh', 0.5) # if cost is below this, then associate
    self.params['do_pruning'] \
      = kwargs.get('do_pruning', True)

  def get_tracked_arrows(self):
    return self.trajectory_set.get_tracked_arrows()

  def save_trajectory_set(self, save_path):
    np.save(save_path, self.trajectory_set.data)

  def reset_db(self):
    self.trajectory_set.reset_db()

  def disable_pruning(self):
    self.params['do_pruning'] = False

  def enable_pruning(self):
    self.params['do_pruning'] = True

  def get_trajectory_set(self):
    return self.trajectory_set

  def update_with_det_in_one_frame(self, arrows1d):
    # assume detections is list of xyxy arrow _2d
    # this function must be called in every frame
    # return: successful association count, newly created trajectories count
    if len(arrows1d) == 0: # here 1d refers to 1D array of arrows
      return 0

    if self.trajectory_set.n_trajectory == 0:
      for arrow in arrows1d:
        self.trajectory_set.add_trajectory(arrow)
      return 0, len(arrows1d)

    # try associate trajectories and detections
    alive_trajectories = self.trajectory_set.get_live_trajectories()
    xyxy_ls_traj = np.array([traj['xyxy'] for traj in alive_trajectories], np.float32)
    xyxy_ls_det = np.array([arrow2d.xyxy for arrow2d in arrows1d], np.float32)

    # calculate cost matrix
    cost_matrix = self.cost_metric.calc_cost_traj_det(xyxy_ls_traj, xyxy_ls_det)

    # hungarian assignment
    traj_indices, det_indices = linear_sum_assignment(cost_matrix)

    matched = []
    unmatched_traj_indicies = set(range(len(alive_trajectories))) # I care only unmatched dets
    unmatched_det_indicies = set(range(len(arrows1d)))

    for ti, di in zip(traj_indices, det_indices):
      if cost_matrix[ti, di] < self.params['do_associate_cost_thresh']:
        matched.append((ti, di))
        unmatched_traj_indicies.discard(ti)
        unmatched_det_indicies.discard(di)

    # merge matched trajectories and detections
    for ti, di in matched:
      arrow = arrows1d[di]
      self.trajectory_set.update_with_arrow2d(ti, arrow)

    # create new trajectories for unmatched detections
    for di in unmatched_det_indicies:
      arrow = arrows1d[di]
      self.trajectory_set.add_trajectory(arrow)

    # pruning
    if self.params['do_pruning']:
      self.trajectory_set.handle_no_updated_trajectory(unmatched_traj_indicies)

    # return number of successful associations and newly created trajectories
    n_successful_associations = len(matched)
    n_new_trajectories = len(unmatched_det_indicies)
    return n_successful_associations, n_new_trajectories

class Detection3DCreaterMediumHole:
  def __init__(self, **kwargs):
    self.valid_labels             = kwargs.get('valid_labels', [])
    self.valid_borders            = kwargs.get('valid_borders', (0, 0, 0, 0))
    self.min_bbox_size            = kwargs.get('min_bbox_size', (0, 0))
    self.max_bbox_size            = kwargs.get('max_bbox_size', (-1, -1))
    self.image_size_wh            = kwargs.get('image_size_wh', (640, 480))
    self.max_nan_ratio            = kwargs.get('max_nan_ratio', 0.4)
    self.depth_unit               = kwargs.get('depth_unit', 'mm')
    self.out_mask_tlbr            = kwargs.get('out_mask_tlbr', (0, 0, 0, 0))
    self.min_dist_from_camera     = kwargs.get('min_dist_from_camera', 0.07)
    self.max_dist_from_camera     = kwargs.get('max_dist_from_camera', 0.5)
    self.growing_medium_target_size = \
      kwargs.get('growing_medium_target_size', (21, 21))
    self.medium_hole_max_size_in_resized_image = \
      kwargs.get('medium_hole_max_size_in_resized_image', int(21**2/4))
    self.medium_hole_border_margin_in_resized_image = \
      kwargs.get('medium_hole_border_margin_in_resized_image', 3)
    self.medium_hole_center_roi_ratio = \
      kwargs.get('medium_hole_center_roi_ratio', 0.1)  # ratio
    self.medium_hole_det_wh_ratio_diff_max = \
      kwargs.get('medium_hole_det_wh_ratio_diff_max', 0.2)
    if self.depth_unit != 'mm':
      raise NotImplementedError(f'depth unit {self.depth_unit} is not implemented')
    else:
      self.depth_unit_factor = 1000.0

    self.validity_checkers = [
      Detection3DCreaterMediumHole.is_valid_label,
    ]
    out_mask_wh = self.out_mask_tlbr[2] - self.out_mask_tlbr[0], \
      self.out_mask_tlbr[3] - self.out_mask_tlbr[1]
    if out_mask_wh[0] > 5 and out_mask_wh[1] > 5:
      self.validity_checkers.append(Detection3DCreaterMediumHole.is_out_of_mask)

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


  def create(self, detection2d, frame, frame_id=''):
    # this function returns (is_valid, measurement)
    # the xyzwlh is in meter, and according to the camera coordinate system
    # check label and bbox

    # assumption: rgb and depth have identical intrinsics
    for checker in self.validity_checkers:
      if not checker(self, detection2d):
        return False, checker.__name__

    depth_np = frame.depth.image
    depth_np = preprocess_depth(depth_np,
                                self.min_dist_from_camera,
                                self.max_dist_from_camera,
                                self.depth_unit)
    depth_K = frame.depth.K
    depth_D = frame.depth.D

    # here, get tlwh with hole detection
    # 1. otsu thresholding
    tlwh = detection2d.tlwh
    rgb_np = frame.rgb.image
    tlx, tly, w, h = int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3])
    ratio_diff = abs(w/h - 1)
    if ratio_diff > self.medium_hole_det_wh_ratio_diff_max:
      return False, f'ratio diff too high: {ratio_diff:.2f}'
    
    roi_size_half = int(w * self.medium_hole_center_roi_ratio), int(h * self.medium_hole_center_roi_ratio)
    best_rect_in_orig = (tlx + w//2 - roi_size_half[0],
                         tly + h//2 - roi_size_half[1],
                         int(roi_size_half[0] * 2),
                         int(roi_size_half[1] * 2))
    ## disable logics due to robustness issues
    # best_rect_in_orig = (tlx, tly, w, h)
    # roi = rgb_np[tly:tly+h, tlx:tlx+w]
    # roi = cv2.resize(roi, self.growing_medium_target_size, interpolation=cv2.INTER_CUBIC)
    # width_ratio = w / self.growing_medium_target_size[1]
    # height_ratio = h / self.growing_medium_target_size[0]

    # gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    # # _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  #  cv2.THRESH_BINARY_INV, 11, 2)
    # _, _, stats, _ = cv2.connectedComponentsWithStats(thresh)
    # filter oversized holes, and find the point closest to the center
    # max_area = -1
    # best_rect = None
    # print('start...', stats)
    # for x, y, w, h, area in stats[1:]:  # skip the background label
    #   if area < self.medium_hole_max_size_in_resized_image:
    #     margin = self.medium_hole_border_margin_in_resized_image
    #     if x <= margin or y <= margin or x + w >= roi.shape[1] - margin or y + h >= roi.shape[0] - margin:
    #       print('margin check error')
    #       continue

    #     if max_area >= area: # cond 1: dist from center
    #       print('area check error')
    #       continue

    #     max_area = area
    #     best_rect = (x, y, w, h)

    # if best_rect is None:
    #   return False, 'no valid hole found'
    # best_rect_in_orig = (int(best_rect[0] * width_ratio + tlx),
    #              int(best_rect[1] * height_ratio + tly),
    #              int(best_rect[2] * width_ratio),
    #              int(best_rect[3] * height_ratio))

    # return True, (best_rect, thresh)
    tlwh = np.array(best_rect_in_orig)
    x, y, w, h = tlwh.astype(int)
    depth_roi = depth_np[y:y+h, x:x+w]
    n_nan = np.isnan(depth_roi).sum()
    n_total = depth_roi.size
    nan_ratio = 1.0*n_nan/n_total
    if nan_ratio > self.max_nan_ratio:
      return False, f'too many nan in depth roi: {nan_ratio*100:.0f}%'

    # calculate dominant depth
    tlwh = detection2d.tlwh
    tlx, tly, w, h = int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3])
    target_ary = depth_np[tly: tly+h, tlx: tlx+w].flatten()
    target_ary = target_ary[~np.isnan(target_ary)]  # remove nans
    depth_roi = set(target_ary)

    tlx, tly, w, h = best_rect_in_orig
    target_ary = depth_np[tly: tly+h, tlx: tlx+w].flatten()
    target_ary = target_ary[~np.isnan(target_ary)]  # remove
    depth_not_interest_roi = set(target_ary)

    depth_interest_set = depth_roi - depth_not_interest_roi
    # depth_interest_set = depth_not_interest_roi
    dominant_depth = np.median(list(depth_interest_set))# + 0.02

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
    sec, nsec = detection2d.ts.sec, detection2d.ts.nsec
    measurement = MfMotDetection3D(sec, nsec, nan_ratio, xyzwlh, frame_id)

    return True, (measurement, best_rect_in_orig, None)
    # return True, (measurement, best_rect_in_orig, thresh)
