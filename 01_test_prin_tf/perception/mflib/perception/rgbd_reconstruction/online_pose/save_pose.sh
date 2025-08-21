#!/bin/bash
python3 subscribe_odometry.py &
PID1=$! 

ros2 launch lidar_to_camera_launch.py &
PID2=$2

cleanup() {
    echo "get SIGINT"
    kill -SIGINT $PID1 2>/dev/null
    kill -SIGINT $PID2 2>/dev/null
    wait $PID1 2>/dev/null
    wait $PID2 2>/dev/null
    echo "program terminated"
    exit 0
}

trap cleanup SIGINT

echo "insert SIGINT when you want to terminate"

wait
echo "program terminated"
