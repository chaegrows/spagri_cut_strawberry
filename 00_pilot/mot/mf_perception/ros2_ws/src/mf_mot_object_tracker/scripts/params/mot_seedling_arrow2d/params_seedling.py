import numpy as np
# task specific params
ENUM_SEEDLING = 'seedling'
ENUM_GROWING_MEDIUM = 'growing_medium'


## shared params
rgb_topic               = '/camera/camera_hand/color/image_rect_raw/compressed'
rgb_info_topic          = '/camera/camera_hand/color/camera_info'
depth_topic             = '/camera/camera_hand/depth/image_rect_raw/compressedDepth'
depth_info_topic        = '/camera/camera_hand/color/camera_info'
rgb_K = None
rgb_D = None
depth_K = None
depth_D = None

## yolo detection
model_pathes = {}
model_pathes[ENUM_SEEDLING] = '/workspace/ai_models/seedling_yolov8m.pt'
model_pathes[ENUM_GROWING_MEDIUM] = '/workspace/ai_models/growing-medium2.pt'


yolo_debug_topic        = '/mf_perception/yolo_debug/compressed'
bbox_topic              = '/mf_perception/bbox_out'
keypoint_topic          = '/mf_perception/keypoint_out'
yolo_depth_debug_topic  = '/mf_perception/yolo_depth_debug/compressed'

## yolo detection
model_path = '/workspace/ai_models/seedling_yolov8m.pt'
# 1: calyx
# 2:
# 3: unripened fruit
# 4: ripened fruit

conf_thresholds = {}
conf_thresholds[ENUM_SEEDLING] = 0.01
conf_thresholds[ENUM_GROWING_MEDIUM] = 0.1 #dhdh
iou_thresh = 0.3
use_cuda = True
publish_yolo_debug = True
verbose_yolo_predict = False
inference_output_type = ['bbox'] # bbox, keypoint

debug_on_depth = True
max_depth_meter = 0.5



# as computation power allows, decrease dt_mot_timer
dt_mot_timer = 0.05
do_print_health_check = False

# data synchronization
image_queue_size = 100 # few seconds
light_data_queue_size = 200

# odometry
base_frame = 'manipulator_base_link'
body_frame = 'camera_hand_depth_optical_frame'
odom_source = 'tf' # 'tf' or 'topic'
# if odom_source == 'topic':
#   odom_topic = '/mavros/odometry/out'


tf_timer_period_seconds = 0.1

max_allowed_rgbd_timediff_ms = 1000 # assume cam is static, so timediff does not matter
max_allowed_odom_time_diff_ms = 1000 # assume cam is static, so timediff does not matter

verbose_sync = False


# 3D object detection
depth_min_mm      = 70
depth_max_mm      = 500
depth_unit        = 'mm'

LABEL_SEEDLING_ROOT = 0
LABEL_SEEDLING_FULL = 1


LABEL_SEEDLING_ROOT_SCORE_THRESH = 0.3
exclusive_pixel_thresh = 20
enable_select_most_likely_seedling = False
arrow_min_pixel_length_thresh = 10 # pixel
debug_seedling_detection = True

# trajectory management
reset_db_posediff_thresh = 0.01 # meters
seedling_trajectory_params = {
  # trajectory set
  'start_update_w': 0.5,  # weight for start point update
  'end_update_w': 0.5,    # weight for end point update
  'n_required_association': 3,  # number of detections required to confirm a trajectory
  'pruning_no_update_count_thresh': 5,  # number of frames without update before pruning
  # cost metrics
  'max_allowed_center_dist': 20,  # max allowed center distance for association (in pixels)
  'max_allowed_slope_diff': 0.2617993877991494,  # max allowed slope difference for association (in radians)
  'center_cost_w': 0.5,  # weight for center distance cost
  'angle_cost_w': 0.5,  # weight for angle difference cost
  # trajectory management
  'do_associate_cost_thresh': 0.3,
  'do_pruning': True,
}


# to arrow in 3D
depth_roi_size = 30
debug_arrow_in_3d = True


# growing medium
LABEL_GROWING_MEDIUM_EMPTY = 0

# hole detection
hole_detection_params = {
  'valid_labels': [LABEL_GROWING_MEDIUM_EMPTY],
  'valid_borders': (0, 0, 0, 0),
  'min_bbox_size': (0, 0),
  'max_bbox_size': (-1, -1),
  'image_size_wh': (640, 480),  # width, height
  'max_nan_ratio': 0.4,
  'max_dist_from_camera': 0.5,
  'growing_medium_target_size': (21, 21),
  'medium_hole_max_size_in_resized_image': int(21**2/4),
  'medium_hole_border_margin_in_resized_image': -1,
  'medium_hole_center_roi_ratio': 0.1, # ratio*2 is the length of the roi
  'medium_hole_det_wh_ratio_diff_max': 0.2,  # max allowed ratio difference between width and height of the detected hole
}

debug_growing_medium_holes_image = True
# growing_medium_trajectory_params = {
