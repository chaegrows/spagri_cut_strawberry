# git submodule update --init --recursive (local)

# in docker 
sudo apt update
cd /workspace/third_party/Livox-SDK2/
mkdir build
cd build
cmake .. && make -j
make install
# sudo make install
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/third_party/Livox-SDK2/build/sdk_core/lib" >> ~/.bashrc


cd /workspace/third_party/livox_ws/src/livox_ros_driver2/
./build.sh humble

sleep 3
source /workspace/third_party/livox_ws/install/setup.bash