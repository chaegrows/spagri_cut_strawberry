# 실행방법
- copy_yaml.py 를 통해 원하는 파일을 실행시킵니다. (도커환경을 켜기 전 실행)
```bash
cd /mf_common/config/common
python copy_yaml.py
```
- 사전에 bring_up build 를 해야합니다.
```bash
./env_common_run.sh common rbpodo_ros2 doosan-robot2
./env_common_run.sh {common} {rbpodo_ros2} {doosan-robot2}
# bringup 과 rbpodo_ros2 는 예시 입니다 원하는 launcher ros2 ws 를 import 한 후 version에 따라 docker에 마운트하여 사용하고 싶다면 위와 같이 진행한 후
cd /opt/third_party/rbpodo/build
make install

cd /root/bringup_ws/
colcon build --symlink-install
source install/setup.bash
```

# mf_common 정리


- 공통 소스 패키지
  - 모든 패키지에서 공통으로 사용하는 소스파일을 담는 저장소입니다.

- 브랜치 운영 원칙
  - 기본적으로 dev에 있는 내용을 사용
  - 기능 개선이 필요할 경우, 하위 기능(예: feat_submodule)별로 분기하여 업데이트

    ```
    main  # README 등의 문서
    dev #
      └─ feat_{submodule}
    ```

- 구성 가이드
  - {local_ws} 하위에 각각 원하는 git을 clone
  - **common 은 필수 -> dev branch**
  - **main project clone**
    - 예시
    ```bash

    {local_ws}/
    ├── mf_common/
    |   └── mf_lib/
    |       └── common/
    └── mf_endeffector/
        └── mf_lib/
            └── common/
            └── endeffector/
        └── docker/

    ```
  - mflib.common.을 기본으로, 필요한 하위 모듈은 mflib.{submodule_name} 형태로 폴더 생성
    - ex : mflib.manipulator.piper, mflib.manipulator.doosan, mflib.perception, mflib.endeffector
  - 각 도커는 목표하는 대상의 폴더 내부에서 실행합니다
    - ex ) mf_enndeffector 폴더에서 실행
    ```bash
    mf_endeffector/
    └── mf_lib/
        └── common/
        └── endeffector/
    └── docker/

    . ./docker/scripts/docker_build.sh
    . ./docker/scripts/docker_launch.sh

    ```


