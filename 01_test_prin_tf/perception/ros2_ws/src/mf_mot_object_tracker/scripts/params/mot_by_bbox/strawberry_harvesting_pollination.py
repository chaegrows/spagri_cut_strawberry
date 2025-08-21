import numpy as np

## shared params
rgb_topic               = '/camera/camera_hand/color/image_rect_raw/compressed'
rgb_info_topic          = '/camera/camera_hand/color/camera_info'
depth_topic             = '/camera/camera_hand/depth/image_rect_raw'
depth_info_topic        = '/camera/camera_hand/depth/camera_info'
rgb_K = None
rgb_D = None
depth_K = None
depth_D = None

yolo_debug_topic        = '/mf_perception/yolo_debug/compressed'
bbox_topic              = '/mf_perception/bbox_out'
keypoint_topic          = '/mf_perception/keypoint_out'
yolo_depth_debug_topic  = '/mf_perception/yolo_depth_debug/compressed'

#

## yolo detection
model_path = '/workspace/ai_models/straw_keypoint.pt' # for x86_64
# model_path = '/workspace/ai_models/strawberry2024DB.pt' # for x86_64
# model_path = '/workspace/ai_models/strawberry2024DB.engine' # for aarch64 (jetson)
use_cuda = True
# 1: calyx
# 2:
# 3: unripened fruit
# 4: ripened fruit

conf_thresh = 0.1
iou_thresh = 0.1
publish_yolo_debug = True #cycy
verbose_yolo_predict = True
print_labels = True
inference_output_type = ['bbox', 'keypoint'] # bbox, keypoint

debug_on_depth = True
max_depth_meter = 0.5



# as computation power allows, decrease dt_mot_timer
dt_mot_timer = 0.05


# data synchronization
image_queue_size = 500 # few seconds
light_data_queue_size = 700

# odometry
base_frame = 'camera_hand_link'
body_frame = 'camera_hand_depth_optical_frame'
odom_source = 'tf' # 'tf' or 'topic'
# if odom_source == 'topic':
#   odom_topic = '/mavros/odometry/out'

# mot params
print_mot_health_check = True


tf_timer_period_seconds = 0.05

max_allowed_rgbd_timediff_ms = 10
max_allowed_odom_time_diff_ms = 200

verbose_sync = True


# 3D object detection
depth_min_mm      = 70
depth_max_mm      = 700
depth_unit        = 'mm'

det3d_params = {
  'valid_labels': [5, 7], # unripened
  # 'valid_labels': [5], # unripened
  'valid_borders': (0, 0, 0, 0),
  'min_bbox_size': (0, 0),
  'max_bbox_size': (320, 240),
  'image_size_wh': (640, 480),
  'out_mask_tlbr': (0,0,0,0),
  'max_nan_ratio': 0.4,
  'max_dist_from_camera': depth_max_mm/1000.,
}

detection3d_verbose = True
debug_det3d = True

labels_to_names_and_colors = {
  0: {'name': 'L0', 'color': (0, 255, 0)},
  1: {'name': 'L1', 'color': (0, 165, 255)},
  2: {'name': 'L2', 'color': (255, 0, 0)},
  3: {'name': 'L3', 'color': (0, 255, 0)},
  4: {'name': 'L4', 'color': (130, 0, 75)},
  5: {'name': 'L5', 'color': (238, 130, 238)},
  6: {'name': 'L6', 'color': (0, 0, 0)},
  7: {'name': 'L7', 'color': (127, 127, 127)},
  8: {'name': 'L8', 'color': (255,255, 255)},
  9: {'name': 'L9', 'color': (255,255, 255)},
  10: {'name': 'L10', 'color': (255,255, 255)},
  11: {'name': 'L11', 'color': (255,255, 255)},
  12: {'name': 'L12', 'color': (255,255, 255)},
  13: {'name': 'L13', 'color': (255,255, 255)},
  14: {'name': 'L14', 'color': (255,255, 255)},
  15: {'name': 'L15', 'color': (255,255, 255)},
  16: {'name': 'L16', 'color': (255,255, 255)},
  17: {'name': 'L17', 'color': (255,255, 255)},
  18: {'name': 'L18', 'color': (255,255, 255)},
  19: {'name': 'L19', 'color': (255,255, 255)},
}

# association and lifecycle management
# cost_metric_params = {
#   'use_cube_iou': True,
#   'weight_cube_iou': 0.3,
#   'use_cube_dist_l2': True,
#   'weight_cube_dist_l2': 0.02,
#   'use_cube_dist_size_l2': False,
#   'weight_cube_dist_size_l2': 0,
# } #strawberry

cost_metric_params = {
  # sum of weights should be 1
  'use_cube_iou': False,
  'weight_cube_iou': 0.3,
  'use_cube_dist_l2': True,
  'weight_cube_dist_l2': 1.0,
  'max_allowed_cube_dist_l2': 0.02,
  'use_cube_dist_size_l2': False,
  'weight_cube_dist_size_l2': 0,
} #strawberry

trajectory_params = {
  'verbose': False,
  'n_max_trajectory': 10000,
  'pose_update_w': 0.7,
  'volume_update_w': 0.7,
  'det2d_update_factor': 0.5,
  'det2d_max_score': 0.5,
  'n_required_association': 3,
  'thresh_confirmation': 0.3,
  'query_xyz_radius': 0.1,
  'n_max_neighbor': 10,
  'do_associate_cost_thresh': 1.0, # should be 0 ~ 1
  'do_pruning': True,
  'pruning_no_update_count_thresh': 5,
}

debug_trajectory_mode = 'verbose' # None, 'simple', 'verbose'
debug_trajectory_min_det3d_count = 3

# other utils
use_manipulator_adapter = True
