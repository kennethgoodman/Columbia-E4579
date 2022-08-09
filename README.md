# Columbia-E4579
Fall 2022 Class At Columbia. Modern Recommendation Systems

## Local Dev Setup

### Set Up Python Env
First open a terminal and create a virtual environment

#### WARNING IF USING M1
If using M1, create virtual env with https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706. 
You will also need to update to 12.2+

If not in Conda:

```bash
$ python3 -m venv E4579  
$ source E4579/bin/activate
```

If in Conda:
```bash
$ conda create --name E4579 python=3.8
$ conda activate E4579
$ conda install -c apple tensorflow-deps
$ conda install -c conda-forge mysql
```


Then install flask dependencies (For macOS users, we have the first line to install tensorflow for macOS)
```bash
$ pip install tensorflow-macos tensorflow-metal
$ pip install -r requirements.txt
```

### Setup flask:

You can also add these to an .env file at the top of the repo
```bash
$ export FLASK_APP=project
$ export FLASK_DEBUG=1
```

### Init SQLite

We will not open up a python shell and create the database locally
```bash
$ python3
```

Run these commands in python one time to create the local DB

```python
from project import db, create_app
from project.data_models import _tables

app = create_app()
db.create_all(app=app)
# pass the create_app result so Flask-SQLAlchemy gets the configuration.
```
You should see a db.sqlite file

### Building ReactJS
This will run a react server. In production, we will use a static build, but for development we
want to have a separate server, so we can have hot loading of dev files with auto-reloading.
```bash
$ cd frontend
$ npm install i
$ yarn start
```

### Running app
We can now run flask in another terminal with:
```bash
$ flask run
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