pip install virtualenv
virtualenv ./
source bin/activate
pip install --no-cache-dir -r requirements.txt
deactivate
sudo yum install -y iproute
sudo yum install -y jq
sudo yum install -y lsof