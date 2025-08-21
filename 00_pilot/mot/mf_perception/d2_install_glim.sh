sudo apt install curl gpg
curl -s https://koide3.github.io/ppa/setup_ppa.sh | sudo bash
curl -s --compressed "https://koide3.github.io/ppa/ubuntu2204/KEY.gpg" | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/koide3_ppa.gpg >/dev/null
echo "deb [signed-by=/etc/apt/trusted.gpg.d/koide3_ppa.gpg] https://koide3.github.io/ppa/ubuntu2204 ./" | tee /etc/apt/sources.list.d/koide3_ppa.list

sudo apt update
sudo apt install -y libiridescence-dev libboost-all-dev libglfw3-dev libmetis-dev
sudo apt install -y libgtsam-points-cuda12.6-dev 
sudo apt install -y ros-humble-glim-ros-cuda12.6

sudo apt install -y ros-humble-pcl-ros*
sudo rm -rf /usr/lib/python3/dist-packages/blinker*
pip install blinker==1.9.0
pip install open3d