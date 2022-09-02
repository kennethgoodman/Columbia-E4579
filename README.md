# Columbia-E4579
Fall 2022 Class At Columbia. Modern Recommendation Systems

## Local Mac Dev Setup
If you have a windows or linux, the support is not great at the moment. 
Your goal should be to install conda, tensorflow and pip install requirements.

First run the first script to install what you need
```bash
bash scripts/mac_install.sh
conda init bash zsh 
```

Then you need to close the browser once you init conda to be used within bash

In the new terminal run:
```bash
conda activate E4579
bash scripts/install_and_build.sh
```

You can add the "-y" to say yes to everything. like:
```bash
bash scripts/install_and_build.sh -y
```

Note: If you get an error that flask-sqlalchemy or flask-login doesn't exist, you have two options:
1. uninstall flask locally, quit your local venv by running `deactivate` or opening a new shell
```bash
pip uninstall flask
```
2. install flask-sqlalchemy and flask-login globally, quit your local venv by running `deactivate` or opening a new shell
```bash
pip install flask-sqlalchemy flask-login
```

## EC2 Ubuntu Production Server Setup

### Installing python3-venv, npm, nginx and mysql

You can also clone your own fork. Your server_domain_or_IP is the EC2 ip address
```bash
$ git clone https://github.com/kennethgoodman/Columbia-E4579.git
$ sudo bash ./Columbia-E4579/scripts/ec2_ubuntu_install.sh server_domain_or_IP
```

This will end up by opening a .env file, paste the follow and fill out the right side of equal signs:
```text
aws_db_password=
aws_db_endpoint=
aws_db_username=
aws_db_port=
aws_db_schema=E4579
use_aws_db=1
FLASK_APP=project
FLASK_DEBUG=1
SQLALCHEMY_TRACK_MODIFICATIONS=False
use_picsum=1
```

You can now go to server_domain_or_IP and see the website. See server_domain_or_IP/ping on your local browser

To debug problems with the server
```bash
$ sudo systemctl status E4579
```

To debug problems with nginx:
```bash
$ systemctl status nginx.service
```

## Credit
1. Thank you to digitalocean for [tutorial on flask auth](https://www.digitalocean.com/community/tutorials/how-to-add-authentication-to-your-app-with-flask-login)
2. [SamuelSacco](https://github.com/SamuelSacco) for creating the frontend
3. Thank you to Kim Huiyeon for their [tutorial on nginx](https://medium.com/techfront/step-by-step-visual-guide-on-deploying-a-flask-application-on-aws-ec2-8e3e8b82c4f7)
4. Thank you to Andrew Hyndman for their [tutorial on hotloading](https://ajhyndman.medium.com/hot-reloading-with-react-and-flask-b5dae60d9898)