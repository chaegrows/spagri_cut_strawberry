import os
import yaml
import numpy as np

class FarmConfigGenerator:
    def __init__(self, name):
        self.name = name
        self.pcd_map_file = 'isu_room2_250604.pcd'
        self.pcd_viz_map_file = 'isu_room2_250604.pcd'
        self.racks = {}
        self.sectors = []
        self.lanes = []

    def add_rack(self,
                rack_id: str,
                rack_start_xyz: list[float],
                rack_end_xyz: list[float],
                n_horizon: int,
                n_vertical: int,
                plant_offsets: list[float],
                ws_wlh: list[float],
                ws_offset: list[float] = [0.0, 0.0, 0.25]):

        step_horizon = round((rack_end_xyz[0] - rack_start_xyz[0]) / (2*n_horizon), 2)
        step_vertical = round((rack_end_xyz[2] - rack_start_xyz[2]) / n_vertical, 2)

        # 시작점과 끝점을 제외한 중간값들만 생성
        horizon_poses = np.round(np.linspace(rack_start_xyz[0] + step_horizon, rack_end_xyz[0] - step_horizon, n_horizon), 2)
        vertical_poses = np.round(np.linspace(rack_start_xyz[2], rack_end_xyz[2], n_vertical), 2)

        # numpy 연산으로 ws_list_xyzwlh 계산
        h_mesh, v_mesh, p_mesh = np.meshgrid(horizon_poses, vertical_poses, plant_offsets)

        # x, y, z 좌표 계산
        x_coords = h_mesh.flatten() + ws_offset[0]
        y_coords = np.full_like(x_coords, rack_start_xyz[1]) + p_mesh.flatten() + ws_offset[1]
        z_coords = v_mesh.flatten() + ws_offset[2]

        # w, l, h 값 반복
        w_values = np.full_like(x_coords, ws_wlh[0])
        l_values = np.full_like(x_coords, ws_wlh[1])
        h_values = np.full_like(x_coords, ws_wlh[2])

        # 모든 값을 하나의 배열로 결합
        ws_array = np.column_stack((
            np.round(x_coords, 2),
            np.round(y_coords, 2),
            np.round(z_coords, 2),
            np.round(w_values, 2),
            np.round(l_values, 2),
            np.round(h_values, 2)
        ))

        ws_list_xyzwlh = ws_array.tolist()

        rack = {
            'rack_id': rack_id,
            'rack_start_xyz': list(rack_start_xyz),
            'rack_end_xyz': list(rack_end_xyz),
            'n_horizon': n_horizon,
            'n_vertical': n_vertical,
            'horizon_poses': horizon_poses.tolist(),
            'vertical_poses': vertical_poses.tolist(),
            'plant_offsets': list(plant_offsets),
            'ws_list_xyzwlh': [list(x) for x in ws_list_xyzwlh],
        }
        if rack_id in self.racks:
          raise ValueError(f"Rack ID {rack_id} already exists")
        self.racks[rack_id] = rack

    # TODO: 추 후 job_manager에서 계산하도록 수정 필요(farm, robot, work가 모두 고려된 robot의 작업 단위 정의임 = job)
    def calc_robot_work_sectors(self,
              rack_id: str,
              dir_from_rack_to_plants: str,
              robot_ws_depth: float,
              robot_ws_vertical: float,
              robot_ws_horizon: float,
              offset_horizon=0.20,
              offset_vertical=0.0,
              ):

      rack_start_xyz = self.racks[rack_id]['rack_start_xyz']
      rack_end_xyz = self.racks[rack_id]['rack_end_xyz']
      n_vertical = self.racks[rack_id]['n_vertical']
      vertical_poses = np.array(self.racks[rack_id]['vertical_poses'])
      horizon_poses = np.array(self.racks[rack_id]['horizon_poses'])
      offset_from_rack_to_plants = abs(self.racks[rack_id]['plant_offsets'][0])

      if dir_from_rack_to_plants == 'left':
        required_viewpoint = 'right'
        sign = +1
      elif dir_from_rack_to_plants == 'right':
        required_viewpoint = 'left'
        sign = -1
      else:
        raise ValueError(f"Invalid dir_from_rack_to_plants: {dir_from_rack_to_plants}")

      lane_home_pos = [
        round(float(rack_start_xyz[0]), 2), # x in farm frame
        round(float(rack_start_xyz[1] + sign * (robot_ws_depth + offset_from_rack_to_plants)), 2), # y in farm frame
        0.0 # rz in farm frame
      ]
      lane_end_pos = [
        round(float(rack_end_xyz[0]), 2), # x in farm frame
        round(float(rack_end_xyz[1] + sign * (robot_ws_depth + offset_from_rack_to_plants)), 2), # y in farm frame
        0.0 # rz in farm frame
      ]

      robot_sector_vertical_poses = np.array([])
      vertical_step = round((rack_end_xyz[2] - rack_start_xyz[2]) / (n_vertical-1), 2)
      n = int(robot_ws_vertical // vertical_step)  # n+1개씩 평균을 낼 개수

      if n == 0:  # n = 0인 경우 vertical_poses 그대로 사용
        robot_sector_vertical_poses = vertical_poses
      else:
        # n+1개씩 평균을 내고 마지막 남은 것들은 그들끼리 평균
        n_poses = len(vertical_poses)
        n_groups = n_poses // (n + 1)
        remaining = n_poses % (n + 1)

        # n+1개씩 평균 계산
        robot_sector_vertical_poses = np.array([
            np.mean(vertical_poses[i:i+n+1])
            for i in range(0, n_groups * (n + 1), n + 1)
        ])

        # 남은 것들 평균 계산
        if remaining > 0:
            last_group_mean = np.mean(vertical_poses[n_groups * (n + 1):])
            robot_sector_vertical_poses = np.append(robot_sector_vertical_poses, last_group_mean)

      # offset_vertical 적용
      robot_sector_vertical_poses = robot_sector_vertical_poses + offset_vertical

      robot_sector_horizon_poses = np.array([])
      horizon_length = rack_end_xyz[0] - rack_start_xyz[0]
      robot_horizon_step = round(horizon_length / robot_ws_horizon + 1e-10)  # 부동소수점 오차 보정
      robot_sector_horizon_poses = np.arange(rack_start_xyz[0]+offset_horizon, rack_end_xyz[0]-offset_horizon, robot_ws_horizon)

      lane = {
        'rack_id': rack_id,
        'dir_from_rack_to_plants': dir_from_rack_to_plants,
        'lane_home': lane_home_pos,
        'lane_end': lane_end_pos,
      }
      self.lanes.append(lane)


      for v_idx, sector_vertical_pose in enumerate(robot_sector_vertical_poses):
        for h_idx, sector_horizon_pose in enumerate(robot_sector_horizon_poses):

          sector = {
            'rack_id': rack_id,
            'dir_from_rack_to_plants': dir_from_rack_to_plants,
            'horizon_id': h_idx+1,
            'height_id': v_idx+1,
            'global_gv_pose': [round(float(sector_horizon_pose), 2), round(float(lane_home_pos[1]), 2), 0.0],
            'global_height': round(float(sector_vertical_pose), 2),
            'required_viewpoint': required_viewpoint,
            'required_flipped': False,
            'depth_to_plants': round(float(robot_ws_depth), 2),
          }
          self.sectors.append(sector)
          flipped_sector = sector.copy()
          flipped_sector['required_flipped'] = True
          self.sectors.append(flipped_sector)


    def dict_to_list(self, dict_data):
        result = []
        for data in dict_data.values():
            data_copy = dict(data)
            for k, v in data_copy.items():
                if isinstance(v, np.ndarray):
                    data_copy[k] = v.tolist()
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
                    data_copy[k] = [x.tolist() for x in v]
            result.append(data_copy)
        return result

    def generate_yaml(self):
        # numpy array를 모두 list로 변환
        racks = self.dict_to_list(self.racks)

        config = {
            'name': self.name,
            'pcd_map_file': self.pcd_map_file,
            'pcd_viz_map_file': self.pcd_viz_map_file,
            'racks': racks,
            'sectors': self.sectors,
            'lanes': self.lanes
        }

        os.makedirs(f'{self.name}', exist_ok=True)
        yaml_path = f'{self.name}/def.yaml'

        # 앵커 참조를 비활성화하는 Dumper 설정
        class NoAnchorDumper(yaml.Dumper):
            def ignore_aliases(self, data):
                return True

        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, Dumper=NoAnchorDumper, default_flow_style=False, sort_keys=False, allow_unicode=True, width=80)

    def print_summary(self):
        print("\n=== Rack Summary ===")
        print(f"{'Rack ID':<10} {'Start XYZ':<30} {'End XYZ':<30} {'N Horizon':<10} {'N Vertical':<10} {'Plant Offsets':<20} {'WS WLH':<20}")
        print("-" * 130)
        for rack in self.racks.values():
            print(f"{rack['rack_id']:<10} {str(rack['rack_start_xyz']):<30} {str(rack['rack_end_xyz']):<30} {rack['n_horizon']:<10} {rack['n_vertical']:<10} {str(rack['plant_offsets']):<20} {str(rack['ws_list_xyzwlh'][0][3:6]):<20}")
            print(f"{'':>10} Horizon Poses: {str(rack['horizon_poses'])}")
            print(f"{'':>10} Vertical Poses: {str(rack['vertical_poses'])}")
            print()

        print("\n=== Lane Summary ===")
        print(f"{'Rack ID':<10} {'Direction':<15} {'Home Pos':<30} {'End Pos':<30}")
        print("-" * 85)
        for lane in self.lanes:
            print(f"{lane['rack_id']:<10} {lane['dir_from_rack_to_plants']:<15} {str(lane['lane_home']):<30} {str(lane['lane_end']):<30}")

        print("\n=== Sector Summary ===")
        print(f"{'ID':<15} {'Rack ID':<10} {'Direction':<15} {'Horizon ID':<12} {'Height ID':<12} {'Viewpoint':<10} {'Flipped':<8} {'GV Pose':<30}")
        print("-" * 112)
        for sector in self.sectors:
            sector_id = f"l{sector['rack_id']}h{sector['height_id']}h{sector['horizon_id']}r{sector['required_viewpoint'][0]}f{int(sector['required_flipped'])}"
            print(f"{sector_id:<15} {sector['rack_id']:<10} {sector['dir_from_rack_to_plants']:<15} {sector['horizon_id']:<12} {sector['height_id']:<12} {sector['required_viewpoint']:<10} {str(sector['required_flipped']):<8} {str(sector['global_gv_pose']):<30}")


if __name__ == "__main__":
    farm_name = 'isu2025'
    generator = FarmConfigGenerator(name=farm_name)


    if farm_name == 'dongtan2024':

      plant_offsets = [-0.4, 0.4]
      ws_wlh = [1.1, 0.6, 0.4]

      generator.add_rack(
          rack_id=1,
          rack_start_xyz=[2.5, 2.0, 0.8],
          rack_end_xyz=[12.5, 2.0, 2.6],
          n_horizon=8,
          n_vertical=4,
          plant_offsets=plant_offsets,
          ws_wlh=ws_wlh)

      generator.add_rack(
          rack_id=2,
          rack_start_xyz=[2.5, 3.5, 0.8],
          rack_end_xyz=[12.5, 3.5, 2.6],
          n_horizon=8,
          n_vertical=4,
          plant_offsets=plant_offsets,
          ws_wlh=ws_wlh)

      generator.add_rack(
          rack_id=3,
          rack_start_xyz=[2.5, 5.0, 0.75],
          rack_end_xyz=[12.5, 5.0, 2.75],
          n_horizon=8,
          n_vertical=5,
          plant_offsets=plant_offsets,
          ws_wlh=ws_wlh)

      generator.add_rack(
          rack_id=4,
          rack_start_xyz=[2.5, 6.5, 0.75],
          rack_end_xyz=[12.5, 6.5, 2.75],
          n_horizon=8,
          n_vertical=5,
          plant_offsets=plant_offsets,
          ws_wlh=ws_wlh)

      rack_ids = [1, 2, 3, 4]
      robot_ws_depth = 0.3
      robot_ws_vertical = 0.7
      robot_ws_horizon = 0.4

      for rack_id in rack_ids:
        generator.calc_robot_work_sectors(
          rack_id=rack_id,
          dir_from_rack_to_plants='left',
          robot_ws_depth=robot_ws_depth,
          robot_ws_vertical=robot_ws_vertical,
          robot_ws_horizon=robot_ws_horizon)
        generator.calc_robot_work_sectors(
          rack_id=rack_id,
          dir_from_rack_to_plants='right',
          robot_ws_depth=robot_ws_depth,
          robot_ws_vertical=robot_ws_vertical,
          robot_ws_horizon=robot_ws_horizon)

    elif farm_name == 'isu2025':

      ws_wlh = [1.0, 0.6, 0.3]

      generator.add_rack(
          rack_id=1,
          rack_start_xyz=[1.35, 1.0, 0.8],
          rack_end_xyz=[3.8, 1.0, 3.05],
          n_horizon=2,
          n_vertical=4,
          plant_offsets=[0.0],
          ws_wlh=ws_wlh)

      generator.add_rack(
          rack_id=2,
          rack_start_xyz=[1.35, 2.5, 0.8],
          rack_end_xyz=[3.8, 2.5, 3.05],
          n_horizon=2,
          n_vertical=4,
          plant_offsets=[0.0],
          ws_wlh=ws_wlh)

      robot_ws_depth = 0.5
      robot_ws_vertical = 0.8
      robot_ws_horizon = 0.4

      generator.calc_robot_work_sectors(
        rack_id=1,
        dir_from_rack_to_plants='left',
        robot_ws_depth=robot_ws_depth,
        robot_ws_vertical=robot_ws_vertical,
        robot_ws_horizon=robot_ws_horizon)
      generator.calc_robot_work_sectors(
        rack_id=2,
        dir_from_rack_to_plants='right',
        robot_ws_depth=robot_ws_depth,
        robot_ws_vertical=robot_ws_vertical,
        robot_ws_horizon=robot_ws_horizon)


    generator.print_summary()
    generator.generate_yaml()
