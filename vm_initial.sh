sudo apt-get update


# Installing git and configure
sudo apt-get install git
git config --global user.name "Joseph Kim"
git config --global user.email "clo3olb@gmail.com"
git clone https://github.com/clo3olb/trader_ai.git

# Python Setting
# Note that python is already installed
# install venv
sudo apt install python3.11-venv
python3 -m venv trader_ai

# To activate python venv
source trader_ai/bin/activate

# To deactivate python venv
deactivate




