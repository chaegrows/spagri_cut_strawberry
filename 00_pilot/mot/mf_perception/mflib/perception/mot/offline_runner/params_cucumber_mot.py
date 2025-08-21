#!/usr/bin/env python
import os
import params_mot_base as P

params = P

####### create detections 3D params ########
P.bag_file = '/workspace/dset_preprocessing/sj_cu_2'
P.cam_tf_csv_file = '/workspace/data/odom/odom_sj_cu_2.csv'

# P.START_TIME = 10 # seconds
P.rosbag_topic_set = P.rosbag_topic_sets['femto_bolt']
P.START_TIME = 5
P.GENRATE_FRAME_PROCESSING_DURATION = 5

P.detection2d_model_path = '/workspace/Metafarmers/ai_models/cucumber2024DB_aug.pt'
P.detection2d_score_thresh = 0.1
P.detection2d_iou_thresh = 0.001
P.detection2d_interactive = True

P.frame_generation_verbose = False

P.rosbag_topic_sets = {
  'femto_bolt': {
    'rgb_topic'       : '/femto_bolt/color/image_raw/compressed',
    'rgb_info_topic'  : '/femto_bolt/color/camera_info',
    'depth_topic'     : '/femto_bolt/depth/image_raw/compressedDepth',
    'depth_info_topic': '/femto_bolt/depth/camera_info',
  }
}
P.rosbag_topic_set = P.rosbag_topic_sets['femto_bolt']
##################################### det3d #################

detection3d_verbose = False
detection3d_enable_imshow = False


# debug

# 0: background
# 1: cucumber_harvest
# 2: flower_unknown
# 3: leaves
# 4: cucumber_top
# 5: leaf_stem
# 6: cucumber_growing
# 7: flower_male
# 8: flower_female

# create detection3d params
P.det3d_params = {
  'valid_labels': [0, 1, 2, 3, 4, 5, 7, 8], # label is not useful with tomato2024DB.pt
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

P.trajectory_params = {
  'verbose': True,
  'n_max_trajectory': 10000,
  'pose_update_w': 0.5,
  'volume_update_w': 0.8,
  'det2d_update_factor': 0.5,
  'det2d_max_score': 0.7,
  'n_required_association': 3,
  'thresh_confirmation': 0.4,
}

P.cost_metric_params = {
  'use_cube_iou': True,
  'weight_cube_iou': 0.7,
  'use_cube_dist_l2': True,
  'weight_cube_dist_l2': 0.05,
  'use_cube_dist_size_l2': False,
  'weight_cube_dist_size_l2': 0,
}