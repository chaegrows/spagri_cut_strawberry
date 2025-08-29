import numpy as np

## Topics configuration
# RGB camera topics
rgb_topic = '/camera/color/image_rect_raw/compressed'
rgb_info_topic = '/camera/color/camera_info'

# Depth camera topics  
depth_topic = '/camera/depth/image_rect_raw/compressedDepth'
depth_info_topic = '/camera/depth/camera_info'

# YOLO perception topics
yolo_debug_topic = '/mf_perception/yolo_debug/compressed'
bbox_topic = '/mf_perception/bbox_out'
keypoint_topic = '/mf_perception/keypoint_out'
enhanced_debug_topic = '/mf_perception/yolo_debug_enhanced/compressed'

# Additional debug topics
yolo_depth_debug_topic = '/mf_perception/yolo_depth_debug/compressed'

## Camera and TF frames
base_frame = 'camera_link'
depth_frame = 'camera_depth_optical_frame'
body_frame = 'camera_depth_optical_frame'  # Alternative name for depth_frame

## Depth processing parameters
depth_min_mm = 70
depth_max_mm = 700
depth_unit = 'mm'
max_depth_meter = 0.7  # Maximum depth in meters

## GPU settings
cuda_visible_devices = "3"

## Enhanced image saving configuration
save_enhanced_images = True
enhanced_images_folder = '/root/ros2_ws/src/mf_mot_object_tracker/scripts/enhanced_images'
save_tf_data_csv = True

## Stability parameters for temporal smoothing
smoothing_window = 5  # Temporal smoothing window size
confidence_threshold = 0.7  # Minimum confidence for processing
angle_change_threshold = 15.0  # Max angle change per frame (degrees)
position_change_threshold = 0.05  # Max position change per frame (meters)
outlier_rejection_factor = 2.0  # Factor for outlier rejection

## Tracking parameters for fruit association
tracking_distance_threshold = 0.1  # Max distance to associate fruits (meters)
max_missing_frames = 10  # Max frames before removing a track

## YOLO model parameters
model_path = '/workspace/ai_models/250828_CNU_SPAGRI_keypoints.pt'
use_cuda = True
conf_thresh = 0.01
iou_thresh = 0.1
publish_yolo_debug = True
inference_output_type = ['bbox', 'keypoint']  # bbox, keypoint

## Debug and logging settings
verbose_yolo_predict = True
print_debug_info = True
verbose_sync = True
debug_on_depth = True
detection3d_verbose = True
debug_det3d = True

## Keypoint processing parameters
keypoint_confidence_threshold = 0.3  # Minimum confidence for individual keypoints
min_keypoints_required = 3  # Minimum number of keypoints required

## Timer and synchronization parameters
dt_mot_timer = 0.05
tf_timer_period_seconds = 0.05
max_allowed_rgbd_timediff_ms = 10
max_allowed_odom_time_diff_ms = 200

## Data queue parameters
image_queue_size = 500
light_data_queue_size = 700

## 3D detection parameters
det3d_params = {
  'valid_labels': [0, 1, 2],  # Fruit labels
  'valid_borders': (0, 0, 0, 0),
  'min_bbox_size': (0, 0),
  'max_bbox_size': (320, 240),
  'image_size_wh': (640, 480),
  'out_mask_tlbr': (0, 0, 0, 0),
  'max_nan_ratio': 0.4,
  'max_dist_from_camera': depth_max_mm/1000.,
}

## Label definitions and colors
labels_to_names_and_colors = {
  0: {'name': 'fruit_tip', 'color': (255, 255, 255)},      # White - 끝
  1: {'name': 'fruit_stem', 'color': (0, 255, 255)},      # Yellow - 꼭지
  2: {'name': 'fruit_middle', 'color': (0, 0, 255)},      # Red - 중간
  3: {'name': 'fruit_unripe', 'color': (0, 255, 0)},      # Green - 미숙과
  4: {'name': 'fruit_ripe', 'color': (0, 0, 255)},        # Red - 숙과
  5: {'name': 'fruit_other', 'color': (128, 128, 128)},   # Gray - 기타
}

## Association and tracking parameters
cost_metric_params = {
  'use_cube_iou': False,
  'weight_cube_iou': 0.3,
  'use_cube_dist_l2': True,
  'weight_cube_dist_l2': 1.0,
  'max_allowed_cube_dist_l2': 0.02,
  'use_cube_dist_size_l2': False,
  'weight_cube_dist_size_l2': 0,
}

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
  'do_associate_cost_thresh': 1.0,
  'do_pruning': True,
  'pruning_no_update_count_thresh': 5,
}

## Debug trajectory parameters
debug_trajectory_mode = 'verbose'  # None, 'simple', 'verbose'
debug_trajectory_min_det3d_count = 3
print_mot_health_check = True

## Other utility parameters
use_manipulator_adapter = False
odom_source = 'tf'  # 'tf' or 'topic'