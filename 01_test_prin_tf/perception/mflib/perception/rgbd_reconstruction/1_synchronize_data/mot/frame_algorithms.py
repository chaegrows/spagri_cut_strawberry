from . import common
from . import datatypes
import cv2
import numpy as np
from .tracker import BYTETracker


def preprocess_frame(frame, **kwargs):
  # assume equal or bigger size of depth
  assert frame.rgb.image.shape[0] <= frame.depth.image.shape[0], 'depth image should be bigger or equal to rgb image'
  assert frame.rgb.image.shape[1] <= frame.depth.image.shape[1], 'depth image should be bigger or equal to rgb image'

  depth_unit        = kwargs.get('depth_unit', 'mm') # mm or meter

  rgb = frame.rgb.image
  depth = frame.depth.image
  inf_indicies = (depth > 6000).copy()
  if depth_unit == 'mm':
    depth = depth.astype(float)/1000
  else:
    raise NotImplementedError(f'depth unit {depth_unit} is not implemented')
  depth[inf_indicies] = np.nan

  rgb_K, rgb_D = frame.rgb.K, frame.rgb.D
  depth_K, depth_D = frame.depth.K, frame.depth.D

  # register depth to rgb
  Rt = np.array([[1, 0, 0, 0.015],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
  

  depth = cv2.rgbd.registerDepth(depth_K.astype(float), 
                                  rgb_K.astype(float), 
                                  depth_D.astype(float), 
                                  Rt.astype(float), 
                                  depth.astype(float),
                                (depth.shape[1], depth.shape[0]),
                                 depthDilation=True
                                ) #this sometimes destroys the depth image, but let me debug later...
  depth[np.isnan(depth)] = 0
  depth[np.isinf(depth)] = 0
  depth[depth < 0] = 0

  depth = depth[:rgb.shape[0], :rgb.shape[1]]

  frame.depth.image = depth
  return frame
  # # save
  # print('save images...')
  # cv2.imwrite('/workspace/sangju_240728/rgb.png', rgb)
  # depth_save = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
  # cv2.imwrite('/workspace/sangju_240728/depth.png', depth_save)
  # print('save images...done')

  # create statistics of depth image

class MeasurementCreater:
  def __init__(self, **kwargs):
    self.valid_labels             = kwargs.get('valid_labels', [])
    self.valid_borders            = kwargs.get('valid_borders', (0, 0, 0, 0))
    self.min_bbox_size            = kwargs.get('valid_bbox_size', (0, 0))
    self.max_bbox_size            = kwargs.get('max_bbox_size', (-1, -1))
    self.image_size_wh            = kwargs.get('image_size_wh', (640, 480))
    self.max_nan_ratio            = kwargs.get('max_nan_ratio', 0.4)

    self.validity_checkers = [
      MeasurementCreater.is_valid_label,
      MeasurementCreater.is_inside_border,
      MeasurementCreater.is_valid_bbox_size
    ]
  
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

  def create(self, detection, rgb_np, depth_np, rgb_K, rgb_D):
    # this function returns (is_valid, measurement)
    # the xyzwlh is in meter, and according to the camera coordinate system
    # check label and bbox
    for checker in self.validity_checkers:
      if not checker(self, detection):
        return False, 'invalid label or bbox'
      
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


    # undistort uv
    # check if D is 0 vector
    if not np.all(rgb_D < 0.000001):
      return False, 'D is not 0 vector. Need to implement undistortion'
    
    u = x + w/2
    v = y + h/2
    x_center_meter = (u - rgb_K[0, 2]) * dominant_depth / rgb_K[0, 0]
    y_center_meter = (v - rgb_K[1, 2]) * dominant_depth / rgb_K[1, 1]
    z_center_meter = dominant_depth

    width_meter   = w * dominant_depth / rgb_K[0, 0]
    height_meter  = h * dominant_depth / rgb_K[1, 1]
    length_meter  = (width_meter + height_meter) / 2

    xyzwlh = [x_center_meter, y_center_meter, z_center_meter, width_meter, length_meter, height_meter]

    # create measurement
    sec, nsec = detection.ts.sec, detection.ts.nsec
    rgb_roi = rgb_np[y:y+h, x:x+w]
    # measurement = datatypes.MfMotFruitCountingMeasurement(sec, nsec, nan_ratio, xyzwlh, rgb_roi, depth_roi)
    measurement = datatypes.MfMotFruitCountingMeasurement(sec, nsec, nan_ratio, xyzwlh, None, None) # should be option but i'm lazy...
    
    return True, measurement

  def convert_with_tf(self, measurement, transformation_matrix_4x4):
    assert isinstance(measurement, datatypes.MfMotFruitCountingMeasurement), 'measurement should be MfMotFruitCountingMeasurement'
    # convert xyzwlh
    xyz1 = np.array([measurement.xyzwlh[0], measurement.xyzwlh[1], measurement.xyzwlh[2], 1])
    xyz1 = np.dot(transformation_matrix_4x4, xyz1)
    rot = transformation_matrix_4x4[:3, :3]
    wlh = np.array([measurement.xyzwlh[3], measurement.xyzwlh[4], measurement.xyzwlh[5]])
    wlh = np.abs(np.dot(rot, wlh))
    measurement.xyzwlh = [xyz1[0], xyz1[1], xyz1[2], wlh[0], wlh[1], wlh[2]]
    return measurement
  

class FruitTracker3D:
  class FruitCandidate:
    candidate_id = 0
    def __init__(self, measurement, detection, **kwargs):
      self.measurements = [measurement]
      self.detections = [detection]
      self.xyzwlh = measurement.xyzwlh

      self.pose_update_w    = kwargs.get('pose_update_w', 0.5)
      self.volume_update_w  = kwargs.get('volume_update_w', 0.8)
      self.candidate_id = -1

    def assign_candidate_id(self):
      self.candidate_id = FruitTracker3D.FruitCandidate.candidate_id
      FruitTracker3D.FruitCandidate.candidate_id += 1
    
    def update(self, measurement, detection):
      self.measurements.append(measurement)
      self.detections.append(detection)
      
      # update pose
      xyz = np.array(self.xyzwlh[:3])
      new_xyz = np.array(measurement.xyzwlh[:3])
      self.xyzwlh[:3] = (1-self.pose_update_w)*xyz + self.pose_update_w*new_xyz

      # update volume
      wlh = np.array(self.xyzwlh[3:])
      new_wlh = np.array(measurement.xyzwlh[3:])
      self.xyzwlh[3:] = (1-self.volume_update_w)*wlh + self.volume_update_w*new_wlh

    @property
    def ts(self):
      return self.measurements[-1].ts
  
    def get_candidate_id(self):
      return self.candidate_id
  
    def get_frame_count(self):
      return len(self.measurements)

    def get_most_probable_label(self):
      # get most probable label
      labels = [detection.label for detection in self.detections]
      labels = np.array(labels).astype(int)
      unique_labels, counts = np.unique(labels, return_counts=True)
      max_count_idx = np.argmax(counts)
      return unique_labels[max_count_idx]
    
    def get_intersection_box(self, other_fruit_candidate):
      box1_min = np.array(self.xyzwlh[:3]) - np.array(self.xyzwlh[3:])/2
      box1_max = np.array(self.xyzwlh[:3]) + np.array(self.xyzwlh[3:])/2
      box2_min = np.array(other_fruit_candidate.xyzwlh[:3]) - np.array(other_fruit_candidate.xyzwlh[3:])/2
      box2_max = np.array(other_fruit_candidate.xyzwlh[:3]) + np.array(other_fruit_candidate.xyzwlh[3:])/2
      
      # 교차하는 박스의 최소/최대 좌표 계산
      inter_min = np.maximum(box1_min, box2_min)
      inter_max = np.minimum(box1_max, box2_max)
      if np.any(inter_max < inter_min):
        return None
    
      inter_center = (inter_min + inter_max) / 2
      inter_size = inter_max - inter_min
      
      return tuple(inter_center) + tuple(inter_size)
    
    # def get_inverse_iou(self, other_fruit_candidate):
    #   box_overlap = self.get_intersection_box(other_fruit_candidate)
    #   if box_overlap is None:
    #     return 1.0
    #   box1_size = np.array(self.xyzwlh[3:])
    #   box2_size = np.array(other_fruit_candidate.xyzwlh[3:])
    #   box1_volume = np.prod(box1_size)
    #   box2_volume = np.prod(box2_size)
    #   overlap_volume = np.prod(box_overlap[3:])
    #   iou = overlap_volume / (box1_volume + box2_volume - overlap_volume)
    #   return 1.0 - iou

    def getdist__overlap_ratio(self, other_fruit_candidate):
      box_overlap = self.get_intersection_box(other_fruit_candidate)
      if box_overlap is None:
        return 0.0
      box1_size = np.array(self.xyzwlh[3:])
      box1_volume = np.prod(box1_size)
      overlap_volume = np.prod(box_overlap[3:])
      ratio = overlap_volume / box1_volume
      return ratio
  
    def getdist__l2(self, other_fruit_candidate):
      xyz1 = np.array(self.xyzwlh[:3])
      xyz2 = np.array(other_fruit_candidate.xyzwlh[:3])
      return np.linalg.norm(xyz1 - xyz2)
  
    def getdist__non_principle_direction(self, other_fruit_candidiate, principal_direction):
      # ensure size and norm of principal_direction is 3 and 1
      if len(principal_direction) != 3:
        principal_direction = np.array(principal_direction[:3]) # homogeneous coordinate to 3D
      principal_direction = principal_direction / np.linalg.norm(principal_direction)

      xyz1 = np.array(self.xyzwlh[:3])
      cos_th1 = np.dot(xyz1, principal_direction)  # if is is larger than 90 degree? it means FOV is larger than 180,
      non_principal_vec1 = xyz1 - cos_th1 * principal_direction

      xyz2 = np.array(other_fruit_candidiate.xyzwlh[:3])
      cos_th2 = np.dot(xyz2, principal_direction) 
      non_principal_vec2 = xyz2 - cos_th2 * principal_direction

      return np.linalg.norm(non_principal_vec1 - non_principal_vec2)

  
    def __repr__(self):
      return f'FruitCandidate({self.xyzwlh}, id={self.candidate_id}), label={self.get_most_probable_label()}'

  def __init__(self, **kwargs):
    self.fruit_candidates = []

    self.params = {
      'pose_update_w': kwargs.get('pose_update_w', 0.5),
      'volume_update_w': kwargs.get('volume_update_w', 0.8),
      'min_overlap_ratio': kwargs.get('min_overlap_ratio', 0.1),
      'max_allowed_l2_distance': kwargs.get('max_allowed_l2_distance', 0.05),
      'do_merge_cost_max': kwargs.get('do_merge_cost_max', 1.0),
      'just_merge_overlap_ratio': kwargs.get('just_merge_overlap_ratio', 0.3),
      'just_merge_l2_distance': kwargs.get('just_merge_l2_distance', 0.01),
      'candidate_query_rule': kwargs.get('candidate_query_rule', 'last_N'),
      'candidate_query_last_N': kwargs.get('candidate_query_last_N', 10),
      'frame_rate': kwargs.get('frame_rate', 30),
      'BT_track_thresh': kwargs.get('BT_track_thresh', 0.2),
      'BT_track_buffer': kwargs.get('BT_track_buffer', 10),
      'BT_match_thresh': kwargs.get('BT_match_thresh', 0.3),
      'BT_min_box_area': kwargs.get('BT_min_box_area', 10),
      'rgb_image_size_wh' : kwargs.get('rgb_image_size', (640, 480))
    }

    args_tracker = {
      'track_thresh': self.params['BT_track_thresh'],
      'track_buffer': self.params['BT_track_buffer'],
      'match_thresh': self.params['BT_match_thresh'],
      'min_box_area': self.params['BT_min_box_area'],
      'mot20':        False,
    }
    self.tracker = BYTETracker(args_tracker)

  def update_2dtracks(self, detections):
    if len(detections) == 0:
      return []
    
    # decompose msg with label
    preds = []
    for de in detections:
      p = [*de.tlbr, de.score]
      preds.append(p)

    preds = np.array(preds, np.float32)

    # feed to tracker
    online_targets = self.tracker.update(
      preds)
      # preds, [self.params['rgb_image_size_wh'][1], self.params['rgb_image_size_wh'][0],], 
      # [self.rgb_img_height, self.rgb_img_width])
    
    tracks = []

    # map detections and tracks
    for d in detections:
      d_tlwh = d.tlwh
      max_iou = -1
      for t in online_targets:
        t_tlwh = t.tlwh
        # get iou
        tlbr1 = np.array([*d_tlwh[:2], *d_tlwh[:2] + d_tlwh[2:]])
        tlbr2 = np.array([*t_tlwh[:2], *t_tlwh[:2] + t_tlwh[2:]])

        tl = np.maximum(tlbr1[:2], tlbr2[:2])
        br = np.minimum(tlbr1[2:], tlbr2[2:])

        area = np.prod(br - tl) * (tl < br).all()
        iou = area / (np.prod(tlbr1[2:] - tlbr1[:2]) + np.prod(tlbr2[2:] - tlbr2[:2]) - area)

        if iou > max_iou:
          max_iou = iou
          max_iou_track = t
      if max_iou > 0.5:
        track2d = datatypes.MfMot2DImageTrack(
          max_iou_track.tlwh, max_iou_track.track_id, max_iou_track.score)
      else:
        track2d = None
      tracks.append(track2d)
      # print('max_iou:', max_iou)

    sec = detections[0].ts.sec
    nsec = detections[0].ts.nsec
    tracks_mf = datatypes.MfMot2DImageTrackArrayData(sec, nsec, tracks)

    return tracks_mf
  
  def empty(self):
    return len(self.fruit_candidates) == 0
  
  def query_candidates(self, rule):
    if rule == 'last_N':
      n_candidates = len(self.fruit_candidates)
      if n_candidates < self.params['candidate_query_last_N']:
        return self.fruit_candidates, 0
      else:
        return self.fruit_candidates[-self.params['candidate_query_last_N']:], \
          len(self.fruit_candidates) - self.params['candidate_query_last_N']
    else:
      raise NotImplementedError(f'query rule {rule} is not implemented')
    
  def update_one(self, measurement, detection, tracklet, cam_ray_world):
    fruit_candidate = FruitTracker3D.FruitCandidate(
      measurement, detection, 
      pose_update_w=self.params['pose_update_w'],
      volume_update_w=self.params['volume_update_w']
    )

    # if empty, just append
    if self.empty():
      fruit_candidate.assign_candidate_id()
      self.fruit_candidates.append(fruit_candidate)
      return

    # calculate distance to all fruit candidates
    distances = []
    partial_candidates, start_idx = self.query_candidates(self.params['candidate_query_rule'])
    for candidate in partial_candidates:
      # may need to define various distance metrics
      overlap_ratio = candidate.getdist__overlap_ratio(fruit_candidate)
      l2_dist = candidate.getdist__l2(fruit_candidate)
      non_principle_dist = candidate.getdist__non_principle_direction(fruit_candidate, cam_ray_world)

      cost = (1-overlap_ratio) + non_principle_dist
      print(f'overlap ratio, l2_dist, non_principle_dist, cost: \
            {overlap_ratio:.2f}, {l2_dist:.2f}, {non_principle_dist:.2f}, {cost:.2f}')
      # if overlap is large, just merge
      if overlap_ratio > self.params['just_merge_overlap_ratio']:
        distances.append(-1) #negative distance always leads to merge
        continue
      # if non_principle_dist is small, just merge
      if non_principle_dist < self.params['just_merge_l2_distance']:
        distances.append(-1)
        continue
      # if overlap is too small, ignore
      if (overlap_ratio < self.params['min_overlap_ratio']):
        distances.append(None)
        continue
      # if l2 distance is too large, ignore
      if non_principle_dist > self.params['max_allowed_l2_distance']:
        distances.append(None)
        continue
      distances.append(cost)

    # find the closest fruit candidate index
    ## set None to max value
    distances = np.array(distances).astype(float) # None -> np.nan
    nan_indicies = np.isnan(distances)
    if len(distances) == 0 or np.all(nan_indicies):
      fruit_candidate.assign_candidate_id()
      self.fruit_candidates.append(fruit_candidate)
      print('add due to nan...', fruit_candidate.get_candidate_id())
      return
    else:
      distances[nan_indicies] = self.params['do_merge_cost_max'] + 1
      min_dist_idx = np.argmin(distances)
      min_dist = distances[min_dist_idx]
      min_dist_idx_in_fruit_candidates = start_idx + min_dist_idx
      print('min_dist, cost max: ', min_dist, self.params['do_merge_cost_max'])
      if min_dist > self.params['do_merge_cost_max']:
        print('add...')
        fruit_candidate.assign_candidate_id()
        self.fruit_candidates.append(fruit_candidate)
        return
      elif min_dist < self.params['do_merge_cost_max']:
        from_xyz = self.fruit_candidates[min_dist_idx_in_fruit_candidates].xyzwlh[:3].copy()
        self.fruit_candidates[min_dist_idx_in_fruit_candidates].update(measurement, detection)
        to_xyz = self.fruit_candidates[min_dist_idx_in_fruit_candidates].xyzwlh[:3].copy()
        print('update...', self.fruit_candidates[min_dist_idx_in_fruit_candidates].get_candidate_id(),
              f'from:, {from_xyz[0]:.2f}, {from_xyz[1]:.2f}, {from_xyz[2]:.2f}',
              f'to:, {to_xyz[0]:.2f}, {to_xyz[1]:.2f}, {to_xyz[2]:.2f}')
      else:
        raise ValueError('Unexpected control flow')
  def get_reliable_fruit_candidates(self):
    reliable_candidates = []
    for candidate in self.fruit_candidates:
      if candidate.get_frame_count() > 1:
        reliable_candidates.append(candidate)
    return reliable_candidates
    
  def __len__(self):
    return len(self.fruit_candidates)

  def __repr__(self):
    return f'FruitTracker3D({self.fruit_candidates})'  + \
      '\n'.join([f'\t{candidate}' for candidate in self.fruit_candidates])
