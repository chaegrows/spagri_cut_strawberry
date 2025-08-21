#!/bin/bash


if [ $# -lt 1 ]; then
  echo "Usage: $0 [dongtan|office|isu]"
  exit 1
fi

PUB_ENV=$1  # dongtan, office, isu 등
PCD_PATH="" # 아래 case 문에서 설정

# 종료 시 백그라운드 프로세스를 모두 종료
cleanup() {
  echo "🛑 Cleaning up background processes..."
  kill ${GLIM_PID} ${RVIZ_PID} ${PCD_PID} 2>/dev/null || true
}
trap cleanup EXIT

# 에러 발생 시 스크립트 중단
set -e

# third_party 폴더 체크
if [ ! -d "./third_party" ]; then
  echo "📦 'third_party' 폴더가 없어 다운로드를 시작합니다..."
  pip install gdown
  ./download_third_party.sh
else
  echo "📁 'third_party' 폴더가 이미 존재합니다. 바로 시작합니다."
fi

# pcd 폴더 체크
if [ ! -d "./pcd" ]; then
  echo "📦 'pcd' 폴더가 없어 다운로드를 시작합니다..."
  ./download_pcd.sh
else
  echo "📁 'pcd' 폴더가 이미 존재합니다. 바로 시작합니다."
fi

# echo "[1] Building Livox-SDK2..."
# cd /workspace/third_party/
# if [ ! -d "Livox-SDK2" ]; then
#     echo "Cloning Livox-SDK2..."
#     git clone https://github.com/Livox-SDK/Livox-SDK2.git
# fi
# cd Livox-SDK2
# if [ ! -d "build" ]; then
#     mkdir build
# fi
# cd build
# cmake .. && make -j2 && make install

echo "[2] Sourcing glim_ws..."
ros2 run glim_ros glim_rosnode --ros-args -p config_path:=/workspace/third_party/glim/glim_ws/src/glim/config &
GLIM_PID=$!
sleep 2

echo "[3] Sourcing livox_ws..."
cd /workspace/third_party/livox_ws
# if rebuild
if [ ! -d "build" ]; then
    cd src/livox_ros_driver2
    ./build.sh humble
    cd /workspace/third_party/livox_ws
fi

source /workspace/third_party/livox_ws/install/setup.bash
ros2 launch livox_ros_driver2 rviz_MID360_launch.py &
RVIZ_PID=$!
sleep 2

echo "[4] Publishing PCD with pub script..."
case $PUB_ENV in
  "dongtan")
    /opt/mflib/perception/brain/mot/utils/pub_dongtan.sh &
    PCD_PID=$!
    PCD_ARGS_PATH="params_register_before_move_dongtan.py"
    ;;
  "office")
    /opt/mflib/perception/brain/mot/utils/pub_office.sh &
    PCD_PID=$!
    PCD_ARGS_PATH="params_register_before_move.py"
    ;;
  "isu")
    /opt/mflib/perception/brain/mot/utils/pub_isu.sh &
    PCD_PID=$!
    PCD_ARGS_PATH="params_register_before_move_isu.py"
    ;;
  *)
    echo "Unknown environment: $PUB_ENV"
    echo "Use: dongtan, office, isu, etc."
    exit 1
    ;;
esac
sleep 2

if [ -z "$PCD_ARGS_PATH" ]; then
  echo "Error: PCD_ARGS_PATH could not be determined."
  exit 1
fi

echo "[5] Running initial pose registration..."
echo $PCD_ARGS_PATH
# Python 스크립트에 --pcd_map_path 인자를 넘김
python3 /opt/mflib/perception/slam3d/initial_pose/register_before_move.py params_register_before_move.py
  

echo "✅ All processes launched successfully."
