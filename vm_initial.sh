sudo apt-get update

# Installing git and configure
sudo apt-get install git
git config --global user.name "Joseph Kim"
git config --global user.email "clo3olb@gmail.com"
git clone https://github.com/clo3olb/trader_ai.git


# go into the directory(~/trader_ai) and run the following commands

# Python Setting
# Note that python is already installed
# install venv
sudo apt install python3.11-venv
python3 -m venv trader_ai

# To activate python venv
source trader_ai/bin/activate
pip install -r requirements.txt

# For GPU like NVIDIA Tesla V100, You need to install GPU driver
# You may refer to https://github.com/GoogleCloudPlatform/compute-gpu-installation/tree/main/linux
sudo python3 install_gpu_driver.py

# To deactivate python venv
deactivate




