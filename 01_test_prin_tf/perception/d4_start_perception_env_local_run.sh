# git submodule update --init --recursive (local)

# in docker 
# cd /workspace/third_party
# git clone git@github.com:Metafarmers/Livox-SDK2.git
cd /workspace/third_party/Livox-SDK2/
mkdir build
cd build
cmake .. && make -j
sudo make install


cd /workspace/third_party/livox_ws/src/livox_ros_driver2/
./build.sh humble

sleep 3
source /workspace/third_party/livox_ws/install/setup.bash