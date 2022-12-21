#!/bin/bash
sudo yum install git -y
git clone https://github.com/kennethgoodman/Columbia-E4579.git
cd Columbia-E4579/
sudo amazon-linux-extras install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
sudo chkconfig docker on
sudo curl -L https://github.com/docker/compose/releases/download/1.22.0/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose
sudo curl -L https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo amazon-linux-extras install nginx1 -y
sudo amazon-linux-extras install epel -y
sudo yum-config-manager --enable epel -y
sudo yum install git-lfs -y
sudo nano /etc/nginx/nginx.conf
cd Columbia-E4579

sudo /usr/local/bin/docker-compose -f docker-compose.prod.yaml up --build --force-recreate --remove-orphans -d


# add the below under "server.include"
# location /{
#    proxy_pass http://localhost:5004/;
# }

