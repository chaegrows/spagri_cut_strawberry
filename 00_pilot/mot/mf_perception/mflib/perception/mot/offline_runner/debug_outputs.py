import rclpy
import pickle as pkl
from visualization_msgs.msg import Marker, MarkerArray
from mflib.brain.mot.common import Timestamp as cTimestamp

mode = 'detection3d'
detection_out_file = '/workspace/data/mot_out/detection3d/detections3d_000.pkl'

class MarkerPublisher: 
  def __init__(self):
    self.node = rclpy.create_node('marker_publisher')
    self.det3d_publisher = self.node.create_publisher(MarkerArray, 'det3d_cubes', 10)
    self.marker_array = MarkerArray()

  
  def preprocess_det3d(self, det3ds, params):
    self.det3ds = det3ds
    self.det3d_params = {}
    
    self.det3d_params['publish_rule'] = params.get('publish_rule', 'per_frame') # per_frame, per_seconds
    if self.det3d_params['publish_rule'] == 'per_seconds':
      self.det3d_params['per_seconds_unit'] = params.get('seconds', 1)
    self.det3d_params['accumulate'] = params.get('accumulate', True)
    self.det3d_params['wait_rule'] = params.get('wait_rule', 'keyboard_enter') # keyboard_enter, none

    self.det3ds_markers_list = []
    det3ds_markers = MarkerArray()
    base_ts = None
    # last_det3d = None

    print(f'preprocess {len(det3ds)} detection3ds...')
    for m_id, det3d in enumerate(det3ds):
      if base_ts is None: 
        base_ts = det3d.ts
      elif self.det3d_params['publish_rule'] == 'per_frame':
        if base_ts != det3d.ts:
          self.det3ds_markers_list.append(det3ds_markers)
          det3ds_markers = MarkerArray()
          base_ts = det3d.ts
      elif self.det3d_params['publish_rule'] == 'per_seconds':
        # if det3d.ts - last_ts > self.det3d_params['per_seconds_unit']: # this not work. I banned...
        gap_seconds = (det3d.ts - base_ts).toSeconds()
        if gap_seconds >= self.det3d_params['per_seconds_unit']:
          self.det3ds_markers_list.append(det3ds_markers)
          det3ds_markers = MarkerArray()

          base_ts = cTimestamp(base_ts.sec + 1, base_ts.nsec)
      else:
        raise ValueError(f'invalid publish_rule: {self.det3d_params["publish_rule"]}')

      m = Marker()
      m.header.frame_id = "map"
      # m.header.frame_id = det3d.frame_id
      m.header.stamp.sec = det3d.ts.sec
      m.header.stamp.nanosec = det3d.ts.nsec
      m.ns = 'det3d_cubes'
      m.id = m_id
      m.type = Marker.CUBE
      m.action = Marker.ADD
      m.pose.position.x = det3d.xyzwlh[0]
      m.pose.position.y = det3d.xyzwlh[1]
      m.pose.position.z = det3d.xyzwlh[2]
      m.pose.orientation.x = float(0)
      m.pose.orientation.y = float(0)
      m.pose.orientation.z = float(0)
      m.pose.orientation.w = float(1)
      m.scale.x = det3d.xyzwlh[3]
      m.scale.y = det3d.xyzwlh[4]
      m.scale.z = det3d.xyzwlh[5]
      m.color.r = float(1)
      m.color.g = float(0)
      m.color.b = float(0)
      m.color.a = float(0.5)
      det3ds_markers.markers.append(m)


  def publish_det3d_cubes(self):
    for m_array in self.det3ds_markers_list:
      if self.det3d_params['accumulate'] == False:
        m = Marker()
        m.header.frame_id = 'haha'
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.ns = 'det3d_cubes'
        m.id = 0
        m.type = Marker.DELETEALL
        m.action = Marker.DELETEALL
        m_array.markers.append(m)
      self.det3d_publisher.publish(m_array)

      self.det3d_publisher.publish(m_array)
      if self.det3d_params['wait_rule'] == 'keyboard_enter':
        input('type_to_proceed....')
  

def main(args=None):
  rclpy.init(args=args)
  mp = MarkerPublisher()

  if mode == 'detection3d':
    with open(detection_out_file, 'rb') as f:
      det3ds = pkl.load(f)

    params = {
      # 'publish_rule': 'per_frame',
      'publish_rule': 'per_seconds',
      'per_seconds_unit': 1,
      'accumulate': True,
      'wait_rule': 'keyboard_enter' #keyboard_enter
    }
    mp.preprocess_det3d(det3ds, params)
    mp.publish_det3d_cubes()


if __name__ == '__main__':
  main()