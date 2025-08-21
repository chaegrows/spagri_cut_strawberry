import numpy as np

## shared params
rgb_topic               = '/d405/color/image_rect_raw/compressed'
rgb_info_topic          = '/d405/color/camera_info'
depth_topic             = '/d405/depth/image_rect_raw'
depth_info_topic        = '/d405/depth/camera_info'
# rgb_K = np.array([
#   [435.6451110839844, 0, 429.607177734375],
#   [0, 434.5065002441406, 247.03179931640625],
#   [0, 0, 1]
# ])
# rgb_D = np.array([-0.05196697264909744,
#                   0.05684574693441391,
#                   -0.0005328544066287577,
#                   0.0012607411481440067,
#                   -0.01910918578505516
# ])

# depth_K = np.array([
#   [430.1526184082031, 0, 431.3135986328125],
#   [0, 430.1526184082031, 242.7838134765625],
#   [0, 0, 1]
# ])
# depth_D = np.array([0,0,0,0,0])

# rgb_topic               = '/femto_bolt/color/image_raw/compressed'
# rgb_info_topic          = '/femto_bolt/color/camera_info'
# depth_topic             = '/femto_bolt/depth/image_raw'
# depth_info_topic        = '/femto_bolt/depth/camera_info'
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
model_path = '/workspace/ai_models/tomato2024DB.pt'
# 1: calyx
# 2:
# 3: unripened fruit
# 4: ripened fruit

conf_thresh = 0.1
iou_thresh = 0.8
use_cuda = True
publish_yolo_debug = True 
verbose_yolo_predict = False
print_labels = True
inference_output_type = ['bbox'] # bbox, keypoint

debug_on_depth = True
max_depth_meter = 0.5



# as computation power allows, decrease dt_mot_timer
dt_mot_timer = 0.05


# data synchronization
image_queue_size = 100 # few seconds
light_data_queue_size = 200

# odometry 
base_frame = 'map'
body_frame = 'femto_bolt'
odom_source = 'topic' # 'tf' or 'topic'
if odom_source == 'topic':
  odom_topic = '/mavros/odometry/out'



tf_timer_period_seconds = 0.1

max_allowed_rgbd_timediff_ms = 10
max_allowed_odom_time_diff_ms = 200

verbose_sync = False


# 3D object detection
depth_min_mm      = 70
depth_max_mm      = 1000
depth_unit        = 'mm'

det3d_params = {
  # 'valid_labels': [2], # ripened 
  'valid_labels': [3], # unripened 
  'valid_borders': (0, 0, 0, 0),
  'min_bbox_size': (0, 0),
  'max_bbox_size': (50, 50),
  'image_size_wh': (848, 480),
  'out_mask_tlbr': (0,0,0,0), 
  'max_nan_ratio': 0.4,
  'max_dist_from_camera': 2.0
}

#{0: 'background', 
# 1: 'tomato stem', 
# 2: '3rd stage', 
# 3: '1st stage', 
# 4: '2nd stage', 
# 5: 'cluster', 
# 6: 'leaf-node', 
# 7: 'flower'}

detection3d_verbose = False
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
  'max_allowed_cube_dist_l2': 0.05,
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
  'n_required_association': 4,
  'thresh_confirmation': 0.8,
  'query_xyz_radius': 0.1,
  'n_max_neighbor': 10,
  'do_associate_cost_thresh': 1.0, # should be 0 ~ 1
  'do_pruning': False,
  'pruning_no_update_count_thresh': 4,
}

debug_trajectory_mode = 'simple' # None, 'simple', 'verbose'
debug_trajectory_min_det3d_count = 2

# other utils
use_manipulator_adapter = False
