
# Todolist

- [ ] Provide docker build script
- [ ] Support task to seedling, dongtan strawberry farm, isu GUI, ...

## Setting
- Folder 구조는 아래와 같이 설정

```
(Ubuntu 기준)
📁 ~ (최상위 디렉토리, Project Root)
├── 📁 common (mf_common을 다운로드 받고 common으로 이름 변경)
│   ├── 📂 config
│   ├── 📂 docker
│   ├── 📂 third_party
│   └── ...
├── 📁 perception (mf_perception을 다운로드 받고 perception으로 이름 변경)
│   ├── 📂 config
│   ├── 📂 data
│   ├── 📂 docker
│   ├── 📂 mflib
│   └── ...
```

- common과 perception은 같은 상위 디렉터리 내에 위치해야 합니다.

- perception 홈 폴더에서 아래 커맨드로 도커 빌드
```bash
projectroot 기준

cd perception
./docker/ubuntu2204_cuda12_ros2humble/build_image.sh
```

- 도커 실행
```bash
projectroot 기준

cd perception
./env_run.sh
```

## Downloads
```bash
./download_pt.sh # AI model
./download_pcd.sh # map 
```