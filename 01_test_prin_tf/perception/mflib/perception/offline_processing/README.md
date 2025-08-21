# 저수준 mf_perception offline 데이터 정의 규칙
- **모든 메타데이터 포맷은 `h5py`로 통일**
  - 일관된 저장 방식 유지
  - 바이너리 + 메타정보 저장 가능

- **`.h5` 파일을 정의할 때는 자료 구조 설명을 반드시 포함**
  - 각 데이터셋에 `attrs['description']` 필수
  - 계층 구조는 명확하게 작성

- **`.h5` 내부의 배열은 모두 `NumPy` 형식으로 저장**
  - `np.array` 사용

- **이미지 등 대용량 데이터는 별도 파일로 분리 저장**
  - `.h5`에는 해당 경로만 참조로 포함
- **이름 규칙**
  - 복수 접미사 (s, es 등) 사용 X

### DB 폴더 구조 출력 명령어
```
tree -F /workspace/data/db
```
예시
```
/workspace/data/db
|-- mot
|   `-- by_bbox
|       `-- 250408_example_bag
|           |-- det3d
|           `-- track
|-- raw
|   |-- rosbag2
|   |   |-- 250408_example_bag
|   |   `-- 250413_room_815
|   `-- smartphone
|-- source
|   |-- ai_models
|   |   |-- detection
|   |   `-- instance_segmentation
|   |-- sensors
|   |   |-- 250408_example_bag
|   |   |   |-- camera_camera_hand_color_image_rect_raw_compressed
|   |   |   `-- camera_camera_hand_depth_image_rect_raw
|   |   `-- 250413_room_815
|   |       |-- camera_color_image_raw_compressed
|   |       |-- camera_depth_image_raw_compressedDepth
|   |       `-- livox_lidar
|   `-- specs
|       |-- farm
|       |-- robot
|       `-- work
`-- synced
    `-- rosbag2
        `-- 250408_example_bag
```


### ✅ 지원되는 Offline Processing 단계

| 상태  | 단계                            | 설명                                 |
|-------|----------------------------------|--------------------------------------|
| ⏳     | raw rosbag2 → source            | 사용 가능, 추가 개발 진행 중         |
| ▫️     | source/sensors → synced         | 준비 중                               |
| ▫️     | synced → rgbd map               | 준비 중                               |
| ▫️     | synced → pcd map                | 준비 중                               |
| ▫️     | synced → mot/by_bbox           | 준비 중                               |



<details>
  <summary><strong>raw rosbag2 → source 명령어 확인</strong></summary>

  ```bash
    python3 raw_to_source/rosbag2_to_source.py \
    --bag_path /workspace/data/db/raw/rosbag2/250413_room_815/
  ```
</detail>