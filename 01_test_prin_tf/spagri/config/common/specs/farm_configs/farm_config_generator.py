import os
import yaml
import numpy as np

class FarmConfigGenerator:
    def __init__(self, name, pcd_map_file, pcd_viz_map_file):
        self.name = name
        self.pcd_map_file = pcd_map_file
        self.pcd_viz_map_file = pcd_viz_map_file
        self.workspaces = []
        self.sectors = []
        self.lanes = []

    def add_lane(self, lane_id, lane_pos, horizon_range, vertical_range, dir_to_plants, depth_to_plants, ws_wlh):
        horizon_poses = np.arange(horizon_range[0], horizon_range[1] + horizon_range[2], horizon_range[2])
        vertical_heights = np.arange(vertical_range[0], vertical_range[1] + vertical_range[2], vertical_range[2])

        for h_idx, h_pose in enumerate(horizon_poses, 1):
            for v_idx, v_height in enumerate(vertical_heights, 1):
                gv_pose_y = lane_pos + (depth_to_plants if dir_to_plants == 'left' else -depth_to_plants)

                sector = {
                    'lane_id': lane_id,
                    'height_id': v_idx,
                    'horizon_id': h_idx,
                    'required_viewpoint': dir_to_plants,
                    'global_height': round(float(v_height), 2),
                    'global_gv_pose': [round(float(h_pose), 2), round(float(gv_pose_y), 2), 0.0],
                    'depth_to_plants': round(float(depth_to_plants), 2),
                    'configuration': [
                        round(float(h_pose), 2),
                        round(float(gv_pose_y), 2),
                        round(float(v_height), 2),
                        round(float(ws_wlh[0]), 2),
                        round(float(ws_wlh[1]), 2),
                        round(float(ws_wlh[2]), 2)
                    ]
                }
                self.sectors.append(sector)

        # lane 정보 추가
        lane = {
            'lane_id': lane_id,
            'lane_home': [round(float(horizon_range[0]), 2), round(float(lane_pos), 2), 0.0],
            'lane_end': [round(float(horizon_range[1]), 2), round(float(lane_pos), 2), 0.0]
        }
        self.lanes.append(lane)

    def generate_yaml(self):
        config = {
            'name': self.name,
            'pcd_map_file': self.pcd_map_file,
            'pcd_viz_map_file': self.pcd_viz_map_file,
            'sectors': self.sectors,
            'lanes': self.lanes
        }

        # 폴더 생성
        os.makedirs(f'{self.name}', exist_ok=True)

        # YAML 파일 생성
        yaml_path = f'{self.name}/def.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    # 사용 예시
    generator = FarmConfigGenerator(
        name="test",
        pcd_map_file="test.pcd",
        pcd_viz_map_file="test.pcd"
    )
    lane_start_pos=[2.0, 3.5, 5.0, 6.5]

    horizon_range=[3.0, 13.6, 0.4]
    vertical_range=[1.15, 2.35, 0.6]
    depth_to_plants=0.6
    ws_wlh=[0.4, 0.25, 0.2]

    # lane 추가 예시
    for i, lane_pos in enumerate(lane_start_pos):
      i += 1
      generator.add_lane(
          lane_id=i,
          lane_pos=lane_pos,
          dir_to_plants="right",
          depth_to_plants=depth_to_plants,
          horizon_range=horizon_range,
          vertical_range=vertical_range,
          ws_wlh=ws_wlh
      )
      generator.add_lane(
          lane_id=i,
          lane_pos=lane_pos+2*depth_to_plants,
          dir_to_plants="left",
          depth_to_plants=depth_to_plants,
          horizon_range=horizon_range,
          vertical_range=vertical_range,
          ws_wlh=ws_wlh
      )
    generator.generate_yaml()
