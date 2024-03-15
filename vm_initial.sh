

git config --global user.name "Joseph Kim"
git config --global user.email "clo3olb@gmail.com"
git clone https://github.com/clo3olb/trader_ai.git

# Python is already installed in the image
# PIP is already installed in the image
# install venv
sudo apt-get update
sudo apt install python3.11-venv
python3 -m venv trader_ai
source trader_ai/bin/activate

# install packages
pip install -r requirements.txt

# remove ssh fingerprints
# sometimes it gets wrong when ip address of vm changes
rm ~/.ssh/known_hosts 



