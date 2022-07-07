# Columbia-E4579
Fall 2022 Class At Columbia. Modern Recommendation Systems

## Local Dev Setup

### Set Up Python Env
First open a terminal and create a virtual environment
```bash
$ python3 -m venv E4579
$ source E4579/bin/activate
```

Then install flask dependencies and set two bash variables
```bash
$ pip install -r requirements.txt
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
```bash
$ cd frontend
$ npm install i
$ npm run build
```

### Running app
We can now run flask with:
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

```bash
$ sudo apt-get update
$ sudo apt-get install python3-venv npm nginx mysql-server 
```

Install what is necessary for mysql:
```bash
$ sudo apt-get install python3-dev default-libmysqlclient-dev build-essential
```

### Set Up Python Env
Create a virtual environment
```bash
$ python3 -m venv E4579
$ source E4579/bin/activate
```

Then install flask dependencies and set two bash variables
```bash
$ pip install -r requirements.txt
$ export FLASK_APP=project
$ export FLASK_DEBUG=1
```

### Create the .env file
You should create a .env file and add the necessary variables:
```bash
$ sudo vim .env
```

and put the values:
```text
aws_db_password=
aws_db_endpoint=
aws_db_username=
aws_db_port=
aws_db_schema
use_aws_db=1
```

### Build react
Run these npm commands to install packages and build react
```bash
$ npm install i
$ npm run build
```

### Run gunicorn 

To test that we've set up everything correctly, you can run:
```bash
gunicorn -b 0.0.0.0:8000 project:__init__
```

Then in another terminal to test that it is working:
```bash
$ curl localhost:8000/ping
```
Now you should see "pong"

You can close gunicorn terminal as we start to run this in the background:
```bash
$ sudo nano /etc/systemd/system/E4579.service
```

Write to the file:
```text
[Unit]
Description=Gunicorn instance for Columbia-E4579
After=network.target
[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/Columbia-E4579
ExecStart=/home/ubuntu/Columbia-E4579/E4579/bin/gunicorn --workers=3 --bind=0.0.0.0:8000 --log-level=info 'project.__init__:create_app()'
Restart=always
[Install]
WantedBy=multi-user.target
```

Then start systemctl so that on reload and restart the server will always be up
```bash
$ sudo systemctl daemon-reload
$ sudo systemctl start E4579
$ sudo systemctl enable E4579
```

Finally, we use nginx
```bash
$ sudo systemctl start nginx
$ sudo systemctl enable nginx
```

Now we write to the nginx file:
```bash
sudo nano /etc/nginx/sites-available/E4579
```

You should write to the file: (where server_domain_or_IP = IP address)
```text
server {
    listen 80;
    server_name server_domain_or_IP;

    location / {
        include proxy_params;
        proxy_pass proxy_pass http://127.0.0.1:8000;
   }
}
```

ln the file:
```bash
$ sudo ln -s /etc/nginx/sites-available/E4579 /etc/nginx/sites-enabled
```

Then restart nginx
```bash
$ sudo systemctl restart nginx
```

You can now go to server_domain_or_IP and see the website

## Credit
1. Thank you to digitalocean for [tutorial on flask auth](https://www.digitalocean.com/community/tutorials/how-to-add-authentication-to-your-app-with-flask-login)
2. [SamuelSacco](https://github.com/SamuelSacco) for creating the frontend
3. Thank you to Kim Huiyeon for their [tutorial on nginx](https://medium.com/techfront/step-by-step-visual-guide-on-deploying-a-flask-application-on-aws-ec2-8e3e8b82c4f7)