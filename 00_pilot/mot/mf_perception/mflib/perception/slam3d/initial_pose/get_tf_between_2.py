import numpy as np

# Define the two transformation matrices
T1 = np.array([
    [0.03438359, -0.99726026, -0.06549606, 3.85312064],
    [0.99665457, 0.02935319, 0.07627621, 6.86696618],
    [-0.07414472, -0.06789959, 0.99493327, 1.07943292],
    [ 0., 0., 0., 1. ]
  ])

T2 = np.array([
    [0.05019833, -0.99472523, -0.08945299, 3.78583588],
    [0.99375425, 0.04080939, 0.10386093, 6.82460219],
    [-0.09966257, -0.09410794, 0.99056099, 0.89350284],
    [ 0., 0., 0., 1.]
  ])

# Compute the relative transformation T_rel = T2 * T1_inv
T1_inv = np.linalg.inv(T1)
T_rel = np.dot(T2, T1_inv)
print(T_rel)
# extract x y z y p z
from scipy.spatial.transform import Rotation as R
r = R.from_matrix(T_rel[:3, :3])
ypr = r.as_euler('zyx', degrees=True)
xyz = T_rel[:3, 3]
print('x y z: ', xyz)
print('yaw pitch roll: ', ypr)
# extract x y z
