#!/usr/bin/env python
import os
import params_mot_base as P

params = P

####### create detections 3D params ########
P.bag_file = '/workspace/data/sj_3_3_241025'
P.cam_tf_csv_file = '/workspace/data/odom/odom_sj_3_3.csv'

# P.START_TIME = 10 # seconds
P.rosbag_topic_set = P.rosbag_topic_sets['femto_bolt']

P.detection2d_model_path = '/workspace/Metafarmers/ai_models/tomato2024DB.pt'
P.detection2d_score_thresh = 0.01
P.detection2d_iou_thresh = 0.01
P.detection2d_interactive = True

P.frame_generation_verbose = False
##################################### det3d #################


# debug

# 0: background
# 1: tomato stem
# 2: 3rd stage
# 3: 1st stage
# 4: 2nd stage
# 5: cluster
# 6: leaf-node
# 7: flower

# create detection3d params
P.det3d_params = {
  'valid_labels': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14], # label is not useful with tomato2024DB.pt
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