import numpy as np

## shared params
rgb_topic               = '/camera/color/image_rect_raw/compressed'
rgb_info_topic          = '/camera/color/camera_info'
depth_topic             = '/camera/depth/image_rect_raw/compressedDepth'
depth_info_topic        = '/camera/depth/camera_info'

yolo_debug_topic        = '/mf_perception/yolo_debug/compressed'
bbox_topic              = '/mf_perception/bbox_out'
keypoint_topic          = '/mf_perception/keypoint_out'
yolo_depth_debug_topic  = '/mf_perception/yolo_depth_debug/compressed'


## yolo detection 
model_paths = ['/workspace/ai_models/seedling/badge_seg.pt',
               '/workspace/ai_models/seedling/seedlings-pose-v1.pt',]
              #  '/workspace/mf_perception/ai_models/seedling/seedlings_feb24_2025.pt',]
model_labels_to_newlabel={0: {0: 0}, 1: {0: 1}}
# model_labels_to_newlabel[model_idx][model's label] = new label

conf_thresh = [0.2, 0.01]
iou_thresh = [0.01, 0.01]
use_cuda = True
publish_yolo_debug = True 
verbose_yolo_predict = False
print_labels = True
inference_output_type = ['bbox']

debug_on_depth = False
max_depth_meter = 0.3

rgb_K = np.array([
  [389.4936218261719, 0, 312.99285888671875],
  [0, 389.4936218261719, 241.25473022460938],
  [0, 0, 1]
])
rgb_D = np.array([0,0,0,0,0])

depth_K = np.array([
  [389.4936218261719, 0, 312.99285888671875],
  [0, 389.4936218261719, 241.25473022460938],
  [0, 0, 1]
])
depth_D = np.array([0,0,0,0,0])



# data synchronization
image_queue_size = 60 # few seconds
light_data_queue_size = 100

base_frame = 'base_link'
body_frame = 'camera_depth_optical_frame'

tf_timer_period_seconds = 0.1

max_allowed_rgbd_timediff_ms = 10
max_allowed_odom_time_diff_ms = 300

verbose_sync = False


# 3D object detection
depth_min_mm      = 50
depth_max_mm      = 500
depth_unit        = 'mm'

det3d_params = {
  'valid_labels': [0, 1],  # seedling, growth medium
  'valid_borders': (0, 0, 0, 0),
  'min_bbox_size': (0, 0),
  'max_bbox_size': (100, 100),
  'image_size_wh': (640, 480),
  'out_mask_tlbr': (294, 234, 411, 479), 
  'max_nan_ratio': 0.6,
  'max_dist_from_camera': 2.0
}

  # 0: background
  # 1: flower
  # 2: dead leaves
  # 3: stalk node
  # 4: root node
  # 5: crown
  # 6: unripe fruit
  # 7: ripe fruit
  # 8: stem node
  # 9: ripe completion
  # 10: calyx

detection3d_verbose = False
debug_det3d = False

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
  'use_cube_iou': False,
  'weight_cube_iou': 0.1,
  'use_cube_dist_l2': True,
  'weight_cube_dist_l2': 1,
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
  'n_required_association': 5,
  'thresh_confirmation': 0.5,
  'query_xyz_radius': 0.1,
  'n_max_neighbor': 10,
  'do_associate_cost_thresh': 0.02,
  'do_pruning': True,
  'pruning_no_update_count_thresh': 3,
}

debug_trajectory_update = True


# other utils
use_manipulator_adapter = True
