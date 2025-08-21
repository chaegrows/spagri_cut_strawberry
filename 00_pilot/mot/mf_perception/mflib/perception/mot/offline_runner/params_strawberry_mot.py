#!/usr/bin/env python
import os

####### create detections 3D params ########
# read_bag
bag_file = '/workspace/data/241209/yp_on_mobile_robot2_4_forward'
cam_tf_csv_file = '/workspace/data/odom/odom_yp_on_mobile_robot2_4_forward.csv'

START_TIME = 0 # seconds
PROCESSING_DURATION = 5 # seconds

rgb_topic         = '/femto_bolt/color/image_raw/compressed'
rgb_info_topic    = '/femto_bolt/color/camera_info'
depth_topic       = '/femto_bolt/depth/image_raw'
depth_info_topic  = '/femto_bolt/depth/camera_info'

detection2d_model_path = '/workspace/Metafarmers/ai_models/strawberry2024DB.pt'
# detection2d_model_path = '/workspace/Metafarmers/ai_models/tomato241118.pt'
detection2d_score_thresh = 0.05
detection2d_iou_thresh = 0.01
detection2d_interactive = True

frame_generation_verbose = True
frame_output_dir = '/workspace/data/mot_out/frames'
frame_read_dir = '/workspace/data/mot_out/frames'

##############  ####################### det3d #################

N_DETECTION3D_SAVE = 100000
detection3d_output_dir = '/workspace/data/mot_out/detection3d'
detection3d_verbose = True
detection3d_enable_imshow = False 
detection3d_debug_output_dir = os.path.join(detection3d_output_dir, 'debug')

# debug



# 0: background
# 1: tomato stem
# 2: 3rd stage
# 3: 1st stage
# 4: 2nd stage
# 5: cluster
# 6: leaf-node
# 7: flower

############################ MOT

interest_labels = [3, 4, 5]
min_trajectory_score = 0.4

labels_to_names_and_colors = {
  0: {'name': 'L0', 'color': (0, 255, 0)},
  1: {'name': 'L1', 'color': (0, 165, 255)},
  2: {'name': 'L2', 'color': (0, 255, 255)},
  3: {'name': 'crown', 'color': (0, 127, 255)},
  4: {'name': 'unripenned', 'color': (0, 255, 50)},
  5: {'name': 'flower', 'color': (255, 255, 255)},
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

# create detection3d params
cherrytomato_det3d_params = {
  'valid_labels': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
  'valid_borders': (0, 0, 0, 0),
  'min_bbox_size': (0, 0),
  'max_bbox_size': (-1, -1),
  'image_size_wh': (1280, 720),
  'max_nan_ratio': 0.6,
  'depth_min': 0.0,
  'depth_max': 100.0,
  'depth_unit': 'mm',
  'max_dist_from_camera': 2.0
}

########### Trajectory generation params ###########

strawberry_trajectory_params = {
   # manager params 
  'query_xyz_radius': 0.1,
  'n_max_neighbor': 10,
  'do_associate_cost_thresh': 0.7,
  # trajectory params
  'verbose': True,
  'n_max_trajectory': 10000,
  'pose_update_w': 0.5,
  'volume_update_w': 0.8,
  'det2d_update_factor': 0.5,
  'det2d_max_score': 0.7,
  'n_required_association': 5,
  'thresh_confirmation': 0.7,
}

strawberry_cost_metric_params = {
  
  'use_cube_iou': True,
  'weight_cube_iou': 0.7,
  'use_cube_dist_l2': True,
  'weight_cube_dist_l2': 0.05,
  'use_cube_dist_size_l2': False,
  'weight_cube_dist_size_l2': 0,
}

###########################

cherrytomato_trajectory_params = {
  'verbose': True,
  'n_max_trajectory': 10000,
  'pose_update_w': 0.5,
  'volume_update_w': 0.8,
  'det2d_update_factor': 0.5,
  'det2d_max_score': 0.7,
  'n_required_association': 3,
  'thresh_confirmation': 0.7,
}

cherrytomato_cost_metric_params = {
  'use_cube_iou': True,
  'weight_cube_iou': 0.7,
  'use_cube_dist_l2': True,
  'weight_cube_dist_l2': 0.05,
  'use_cube_dist_size_l2': False,
  'weight_cube_dist_size_l2': 0,
}