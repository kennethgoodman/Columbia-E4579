#!/bin/bash
sudo yum install git docker nginx python3-pip -y
git clone https://github.com/kennethgoodman/Columbia-E4579.git
cd Columbia-E4579/

sudo service docker start
sudo chkconfig docker on

sudo systemctl start nginx
sudo systemctl enable nginx
sudo usermod -a -G docker ec2-user

sudo nano /etc/nginx/nginx.conf

# add the below under "server.include"
location /{
    proxy_pass http://localhost:5004/;
} 

sudo systemctl reload nginx

sudo docker build -t e4579 -f Dockerfile.prod .
export DATABASE_URL=mysql://admin:$DB_PASSWORD@$DATABASEURL:3306/E4579
sudo docker run -e DATABASE_URL=$DATABASE_URL -p 5004:5000 --detach --restart=always e4579
