
# Todolist

- [ ] Provide docker build script
- [ ] Support task to seedling, dongtan strawberry farm, isu GUI, ...

## Setting
- Folder 구조는 아래와 같이 설정

```
(Ubuntu 기준)
📁 ~ (최상위 디렉토리, Project Root)
├── 📁 mf_common
│   ├── 📂 config
│   ├── 📂 docker
│   ├── 📂 third_party
│   └── ...
├── 📁 mf_perception
│   ├── 📂 config
│   ├── 📂 data
│   ├── 📂 docker
│   ├── 📂 mflib
│   └── ...
```

- mf_common과 mf_perception 은 같은 상위 디렉터리 내에 위치해야 합니다.

- mf_perception 홈 폴더에서 아래 커맨드로 도커 빌드
```bash
cd mf_perception
./docker/ubuntu2204_cuda12_ros2humble/build_image.sh
```

- 도커 실행
```bash
cd mf_perception
./env_run.sh
```

## Downloads
```bash
./download_pt.sh # AI model
./download_pcd.sh # map 
```