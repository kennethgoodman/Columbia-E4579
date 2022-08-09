if [ -z "$1" ]
  then
    echo "No argument supplied for IP address"
    exit
fi

# update and get all necessary packages from apt-get
sudo apt-get update
sudo apt-get install python3-venv npm nginx mysql-server
sudo apt-get install python3-dev default-libmysqlclient-dev build-essential

# clone and CD
cd Columbia-E4579 || exit
python3 -m venv E4579
source E4579/bin/activate

# pip install setup
pip install -r requirements.txt || exit
pip install tensorflow-cpu --no-cache-dir || exit
export FLASK_APP=project
export FLASK_DEBUG=1

cd project/frontend || exit
npm install i
npm run build

sudo touch /etc/systemd/system/E4579.service
echo "[Unit]
Description=Gunicorn instance for Columbia-E4579
After=network.target
[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/Columbia-E4579
ExecStart=/home/ubuntu/Columbia-E4579/E4579/bin/gunicorn --workers=3 --bind=0.0.0.0:8000 --access-logfile /home/ubuntu/Columbia-E4579/logs_folder/gunicorn-access.log --error-logfile /home/ubuntu/Columbia-E4579/logs_folder/gunicorn-error.log --log-level=info 'project.__init__:create_app()'
Restart=always
[Install]
WantedBy=multi-user.target" | sudo tee /etc/systemd/system/E4579.service

sudo systemctl daemon-reload
sudo systemctl start E4579
sudo systemctl enable E4579

sudo systemctl start nginx
sudo systemctl enable nginx

sudo touch /etc/nginx/sites-available/E4579

echo "server {
    listen 80;
    server_name $1;

    location / {
        include proxy_params;
        proxy_pass http://127.0.0.1:8000;
   }
}" | sudo tee /etc/nginx/sites-available/E4579

sudo ln -s /etc/nginx/sites-available/E4579 /etc/nginx/sites-enabled

sudo systemctl restart nginx

cd /home/ubuntu/Columbia-E4579 || exit
sudo vim .env
