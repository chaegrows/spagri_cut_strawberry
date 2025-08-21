from copy import deepcopy

  
#   offset_translation_target1, _ = self.rosTF.getTF( f"tcp", f"end_effector")
#   offset_translation_target1_true = deepcopy(offset_translation_target1)
#   offset_translation_target1_true[0] = offset_translation_target1[0] # base x direction
#   offset_translation_target1_true[2] = - offset_translation_target1_true[2] # + 0.02 # + cluster.model.state[5] * 0.5
#   offset_translation_target1_true[1] = - (offset_translation_target1[1] + 0.03)   # base y direction
#   offset_translation_target1_true[1] = offset_translation_target1_true[1] * is_opposite

#   self.rosTF.publishTF(self.params['global_frame_id'],
#                         f"strawberrry_{cluster.id_converge}",
#                         cluster.model.state[:3])

#   self.rosTF.publishTF(f"strawberrry_{cluster.id_converge}",
#                         f"target_{cluster.id_converge}",
#                         offset_translation_target1_true)

def pub_target_tf_suseo(rosTF, trajectory, label):
  if label not in [6, 7]: return
  # classify left right
  trans, _ = rosTF.getTF(f"base_link", f"tcp")
  sign_opposite = 1 if trans[1] >= 0.0 else -1

  offset_translation_target1, _ = rosTF.getTF( f"tcp", f"end_effector")
  offset_translation_target1_true = deepcopy(offset_translation_target1)
  offset_translation_target1_true[0] = offset_translation_target1[0] 
  offset_translation_target1_true[2] = - offset_translation_target1_true[2]
  offset_translation_target1_true[1] = -(offset_translation_target1[1] + 0.03) 
  offset_translation_target1_true[1] = offset_translation_target1_true[1] * sign_opposite

  trans = [float(f) for f in trajectory['xyzwlh'][:3]]
  target_id = trajectory['trajectory_id_if_confirmed']
  rosTF.publishTF('base_link',
                        f"strawberrry_{target_id}",
                        trans) 

  rosTF.publishTF( f"strawberrry_{target_id}",
                        f"target_{target_id}",
                        offset_translation_target1_true)
  
def pub_tf_for_manipulator(rosTF, trajectory, base_frame='manipulator_base_link', target_prefix='strawberry'):
  """
  Publishes target TF based on trajectory information
  Args:
      rosTF: ROS2TF instance
      trajectory: trajectory dictionary containing position info
      base_frame: reference frame name
      target_prefix: prefix for target frame name
  """
  trans = [float(f) for f in trajectory['xyzwlh'][:3]]
  target_id = trajectory['trajectory_id_if_confirmed']
  target_frame = f"{target_prefix}_{target_id}"

  rosTF.publishTF(base_frame, target_frame, trans)
  return target_frame
