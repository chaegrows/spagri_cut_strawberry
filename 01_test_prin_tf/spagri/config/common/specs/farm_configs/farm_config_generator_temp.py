import os
import yaml
import numpy as np

class FarmConfigGenerator:
    def __init__(self, name):
        self.name = name
        self.pcd_map_file = f'{self.name}.pcd'
        self.pcd_viz_map_file = f'{self.name}.pcd'
        self.sectors = []
        self.lanes = []

    def add_lane(self, rack_id, rack_start_xyz, rack_end_xyz, n_horizon, n_vertical, dir_to_plants, ws_wlh):

        step_horizon = round((rack_end_xyz[0] - rack_start_xyz[0]) / n_horizon, 2)
        step_vertical = round((rack_end_xyz[2] - rack_start_xyz[2]) / n_vertical, 2)

        horizon_poses = np.arange(rack_start_xyz[0], rack_end_xyz[0] + step_horizon, step_horizon)
        vertical_poses = np.arange(rack_start_xyz[2], rack_end_xyz[2] + step_vertical, step_vertical)

        for h_idx, h_pos in enumerate(horizon_poses, 1):
            for v_idx, v_pos in enumerate(vertical_poses, 1):
                if dir_to_plants == 'left':     sign = +1
                elif dir_to_plants == 'right':  sign = -1
                else:  raise ValueError(f"Invalid direction: dir_to_plants should be 'left' or 'right'")

                sector = {
                    'rack_id': rack_id,
                    'horizon_id': h_idx,
                    'vertical_id': v_idx,
                    'dir_to_plants': dir_to_plants,
                    # 'global_height': round(float(v_pos), 2),
                    # 'global_gv_pose': [round(float(h_pos), 2), round(float(gv_pose_y), 2), 0.0],
                    # 'depth_to_plants': round(float(depth_to_plants), 2),
                    'ws_xyzwlh': [
                        round(float(h_pos), 2),
                        round(float(rack_start_xyz[1]), 2),
                        round(float(v_pos), 2),
                        round(float(ws_wlh[0]), 2),
                        round(float(ws_wlh[1]), 2),
                        round(float(ws_wlh[2]), 2)
                    ]
                }
                self.sectors.append(sector)

        # # lane 정보 추가
        # lane = {
        #     'rack_id': rack_id,
        #     'lane_home': [round(float(horizon_range[0]), 2), round(float(lane_pos), 2), 0.0],
        #     'rack_end': [round(float(horizon_range[1]), 2), round(float(lane_pos), 2), 0.0]
        # }
        # self.lanes.append(lane)

    def generate_yaml(self):
        config = {
            'name': self.name,
            'pcd_map_file': self.pcd_map_file,
            'pcd_viz_map_file': self.pcd_viz_map_file,
            'sectors': self.sectors,
            # 'lanes': self.lanes
        }

        # 폴더 생성
        os.makedirs(f'{self.name}', exist_ok=True)

        # YAML 파일 생성
        yaml_path = f'{self.name}/def.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    # 사용 예시
    generator = FarmConfigGenerator(name="test")
    # rack_start_pos=[2.0, 3.5, 5.0, 6.5]

    # horizon_range=[3.0, 13.6, 0.4]
    # vertical_range=[1.15, 2.35, 0.6]
    # depth_to_plants=0.6
    # ws_wlh=[0.4, 0.25, 0.2]

    # # lane 추가 예시
    # for i, lane_pos in enumerate(rack_start_pos):
    #   i += 1
    #   generator.add_lane(
    #       rack_id=i,
    #       lane_pos=lane_pos,
    #       dir_to_plants="right",
    #       depth_to_plants=depth_to_plants,
    #       horizon_range=horizon_range,
    #       vertical_range=vertical_range,
    #       ws_wlh=ws_wlh
    #   )
    #   generator.add_lane(
    #       rack_id=i,
    #       lane_pos=lane_pos+2*depth_to_plants,
    #       dir_to_plants="left",
    #       depth_to_plants=depth_to_plants,
    #       horizon_range=horizon_range,
    #       vertical_range=vertical_range,
    #       ws_wlh=ws_wlh
    #   )

    rack_start_xyz = [3.0, 2.0, 1.15]
    rack_end_xyz = [13.6, 2.0, 2.35]
    n_horizon = 10
    n_vertical = 4
    dir_to_plants = 'left'
    ws_wlh = [1.0, 0.25, 0.2]
    generator.add_lane(rack_id=1, rack_start_xyz=rack_start_xyz, rack_end_xyz=rack_end_xyz, n_horizon=n_horizon, n_vertical=n_vertical, dir_to_plants=dir_to_plants, ws_wlh=ws_wlh)

    generator.generate_yaml()
