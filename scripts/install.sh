apt-get update
apt-get install npm vim nano -y
npm install -g pm2
python3 -m venv venv
source venv/bin/activate
export BT_AXON_PORT=29443
pip install -r requirements.txt
pip install -e .

