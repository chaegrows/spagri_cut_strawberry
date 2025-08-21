# algorithm
import open3d as o3d # 0.16.0
import numpy as np
from scipy.spatial.transform import Rotation as R

# ros members
import rclpy
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import message_filters
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

# utils and params
import params_register_before_move as P
import sys
import seaborn

# algorithm
# 1. just accumulate the point cloud
# 2. if any condition satisfied, then do the relocalization

  

class RelocBeforeMove(Node):
  def __init__(self):
    super().__init__('relocalization_with_lio')
    self.first_odom = None
    self.all_points = None

    # load pcd
    self.pcd_map = o3d.io.read_point_cloud(P.pcd_map_path)
    print('n points in map:', len(self.pcd_map.points))

    # broadcast transform
    self.static_tf_broadcaster = StaticTransformBroadcaster(self)

    # relocalization variables
    self.n_scan_accumulated = 0

    # synchronized callback
    self.T_map_P = None
    self.last_reloc_time = None
    self.pointcloud_sub = message_filters.Subscriber(self, PointCloud2, P.lidar_topic)
    self.odom_sub = message_filters.Subscriber(self, Odometry, P.odom_topic) 
    self.ts = ApproximateTimeSynchronizer(
        [self.odom_sub, self.pointcloud_sub], 
        queue_size=100,
        slop=0.01,
    ) 
    self.ts.registerCallback(self.sync_callback)

    # publish initial pose in rviz for visualization and wait
    self.initial_pose_pub = self.create_publisher(PoseStamped, 'mf_slam3d_initial_pose', 100)
    self.initial_pose = PoseStamped()
    self.initial_pose.header.frame_id = 'prebuilt_map'
    self.initial_pose.pose.position.x = float(P.init_xyz[0])
    self.initial_pose.pose.position.y = float(P.init_xyz[1])
    self.initial_pose.pose.position.z = float(P.init_xyz[2])
    q = R.from_euler('z', P.init_yaw_deg, degrees=True).as_quat()
    self.initial_pose.pose.orientation.x = float(q[0])
    self.initial_pose.pose.orientation.y = float(q[1])
    self.initial_pose.pose.orientation.z = float(q[2])
    self.initial_pose.pose.orientation.w = float(q[3])
    self.initial_pose_pub.publish(self.initial_pose)    

    # publish camera FOV simply for visualization
    self.camera_fov_pub = self.create_publisher(MarkerArray, 'mf_slam3d_camera_fov', 10)
    self.camera_fov_marker = self.createFOVmarkers()
    self.camera_fov_pub.publish(self.camera_fov_marker)

    
    print('RelocBeforeMove init done. Now run glim and rosbag play')

  def createFOVmarkers(self):
    marker_array = MarkerArray()
    marker = Marker()
    marker.header.frame_id = 'lidar'
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.1
    marker.color.r = 1.0
    marker.color.a = 1.0
    marker.id = 0
    marker.ns = 'camera_fov'
    marker.points = []
    p = Point()
    p.x = 0.
    p.y = 0.
    p.z = 0.
    marker.points.append(p)
    p = Point()
    p.x = 1.
    p.y = -1.
    p.z = 0.
    marker.points.append(p)
    p = Point()
    p.x = 1.
    p.y = 1.
    p.z = 0.
    marker.points.append(p)
    p = Point()
    p.x = 0.
    p.y = 0.
    p.z = 0.
    marker.points.append(p)
    marker_array.markers.append(marker)
    return marker_array
  
  def registration_at_scale(self, acc_scan_in_L, map_in_P, initial, scale):
    acc_scan_in_L = acc_scan_in_L.voxel_down_sample(P.voxel_size_scan * scale)
    map_in_P = map_in_P.voxel_down_sample(P.voxel_size_map * scale) 

    acc_scan_in_L.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    map_in_P.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    result_icp = o3d.pipelines.registration.registration_icp(
      acc_scan_in_L, map_in_P,
      0.5 * scale, initial,
      o3d.pipelines.registration.TransformationEstimationPointToPoint(),
      o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=200, relative_fitness=1e-3, relative_rmse=1e-3)
    )
    return result_icp.transformation, result_icp.fitness
  
  def visualize_numpy(self, pcd_np):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd_np)

    point_cloud.colors = o3d.utility.Vector3dVector(np.ones_like(pcd_np) * [0, 0.5, 1]) # blue
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([point_cloud, axis], window_name="PointCloud Visualization")

  def visualize_np_ary(self, np_ary):
    palette = seaborn.color_palette('Paired', 50)

    to_show = []
    to_show.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0]))
    for idx, ary in enumerate(np_ary):
      point_cloud = o3d.geometry.PointCloud()
      point_cloud.points = o3d.utility.Vector3dVector(ary)

      point_cloud.colors = o3d.utility.Vector3dVector(np.ones_like(ary) * palette[idx]) # blue
      to_show.append(point_cloud)
    o3d.visualization.draw_geometries(to_show, window_name="PointCloud Visualization")
  
  def get_inverse_tf(self, tf_4x4):
    tf_4x4_inv = np.eye(4)
    tf_4x4_inv[:3, :3] = tf_4x4[:3, :3].T
    tf_4x4_inv[:3, 3] = -np.matmul(tf_4x4[:3, :3].T, tf_4x4[:3, 3])
    return tf_4x4_inv

  def sync_callback(self, odom_msg, lidar_msg):
    self.camera_fov_pub.publish(self.camera_fov_marker)

    # odometry: T_imu_map
    # calculate pose diff
    if self.T_map_P is not None:
      return 

    do_reloc = False

    if self.first_odom is None:
      self.first_odom = odom_msg
    else:
      dx = odom_msg.pose.pose.position.x - self.first_odom.pose.pose.position.x
      dy = odom_msg.pose.pose.position.y - self.first_odom.pose.pose.position.y
      dz = odom_msg.pose.pose.position.z - self.first_odom.pose.pose.position.z
      dist = np.sqrt(dx**2 + dy**2 + dz**2)
      if dist > P.do_reloc_pose_diff:
        do_reloc = True
        self.first_odom = odom_msg
    
    if self.n_scan_accumulated > P.do_reloc_scan_accumulate:
      do_reloc = True
      
    if not do_reloc:
      # accumulate point cloud
      points_raw_bytes = np.frombuffer(lidar_msg.data, dtype=np.uint8)
      points_bytes = points_raw_bytes.reshape(-1, lidar_msg.point_step)
      points_bytes = points_bytes[:, :12]  
      lidar_points = points_bytes.view(np.float32).reshape(-1, 3) # N, 3
      # consider fov
      assert P.fov_degree > 180 # sin cos tan...
      radians = np.arctan2(lidar_points[:, 1], lidar_points[:, 0])
      radian_max = P.fov_degree / 180 * np.pi / 2.0
      lidar_points = lidar_points[np.abs(radians) < radian_max]
      # consider min dist
      dists = np.linalg.norm(lidar_points, axis=1)
      lidar_points = lidar_points[dists >= P.min_dist_to_lidar]
      # consider max dist
      dists = np.linalg.norm(lidar_points, axis=1)
      lidar_points = lidar_points[dists <= P.max_dist_to_lidar]
      ## icra: use selective y
      lidar_points = lidar_points[lidar_points[:, 1] > -3]
      
      if self.all_points is None:
        self.all_points = lidar_points
      else:
        self.all_points = np.vstack([self.all_points, lidar_points])
      self.n_scan_accumulated += 1
      
      print('accumulated point cloud size:', self.all_points.shape[0])
    else:
      
      # self.visualize_numpy(self.all_points)
      # all lidar points need to be rotated with below mat
      from scipy.spatial.transform import Rotation
      rot_imu_to_cam = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
      ]).T
      # convert all lidar ponints
      self.all_points = np.matmul(self.all_points, rot_imu_to_cam.T)
      # self.visualize_np_ary([self.all_points, self.pcd_map.points])
      # save all points as pcd
      o3d.io.write_point_cloud('all_points.pcd', o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.all_points)))


      all_points_o3d_in_L = o3d.geometry.PointCloud()
      all_points_o3d_in_L.points = o3d.utility.Vector3dVector(self.all_points) # these are in lidar frame
      all_points_o3d_in_P = self.pcd_map

      T_lidar_P_prior = np.eye(4)
      T_lidar_P_prior[:3, 3] = P.init_xyz
      T_lidar_P_prior[:3, :3] = R.from_euler('z', P.init_yaw_deg, degrees=True).as_matrix()
      
      # coarse to fine
      transformation1, fitness1 = self.registration_at_scale(
        all_points_o3d_in_L, all_points_o3d_in_P, initial=T_lidar_P_prior, scale=5)
      transformation2, fitness2 = self.registration_at_scale(
        all_points_o3d_in_L, all_points_o3d_in_P, initial=transformation1, scale=1)
      print('transformation1:', transformation1)
      print('transformation2:', transformation2)
      print('fitness1:', fitness1, 'fitness2:', fitness2)
      
      # this code need to consider T_imu_lidar, but for now, just assume it is identity
      T_lidar_P = transformation2
      T_imu_map = np.eye(4) # given from odomety topic
      p = odom_msg.pose.pose.position
      T_imu_map[:3, 3] = [p.x, p.y, p.z]
      q = odom_msg.pose.pose.orientation
      r = R.from_quat([q.x, q.y, q.z, q.w])
      T_imu_map[:3, :3] = r.as_matrix()
      T_lidar_imu = np.eye(4)

      T_lidar_map = T_lidar_imu @ T_imu_map
      T_map_P = self.get_inverse_tf(T_lidar_map) @ T_lidar_P
      print('T_map_P:', T_map_P)
      # print tfmat in csv format with 6 digits
      np.savetxt(sys.stdout, T_map_P, delimiter=',', fmt='%.6f')
      
      # broadcast transform
      tf = TransformStamped()
      tf.header.stamp = odom_msg.header.stamp
      tf.header.frame_id = 'prebuilt_map'
      tf.child_frame_id = 'map'
      tf.transform.translation.x = T_map_P[0, 3]
      tf.transform.translation.y = T_map_P[1, 3]
      tf.transform.translation.z = T_map_P[2, 3]
      q = R.from_matrix(T_map_P[:3, :3]).as_quat()
      tf.transform.rotation.x = q[0]
      tf.transform.rotation.y = q[1]
      tf.transform.rotation.z = q[2]
      tf.transform.rotation.w = q[3]
      self.static_tf_broadcaster.sendTransform(tf)
      
      self.T_map_P = T_map_P
      print('broadcasted transform')

      # print xyz
      rpy = R.from_matrix(T_map_P[:3, :3]).as_euler('xyz', degrees=True)

      print(','.join([f"{x:.4f}" for x in [*T_map_P[:3, 3], *rpy]]))


      R_z_neg_90 = np.array([[ 0,  1,  0,  0],
                        [-1,  0,  0,  0],
                        [ 0,  0,  1,  0],
                        [ 0,  0,  0,  1]])
      print(R_z_neg_90 @ T_map_P)


if __name__ == '__main__':
  rclpy.init(args=sys.argv)
  reloc_node = RelocBeforeMove()
  rclpy.spin(reloc_node)
  rclpy.shutdown()

