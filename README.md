# Columbia-E4579
Fall 2022 Class At Columbia. Modern Recommendation Systems

## Dev Setup

First open a terminal and create a virtual environement
```bash
$ python3 -m venv E4579
$ source E4579/bin/activate
```

Then install flask dependencies and set two bash variables
```bash
$ pip install flask flask-sqlalchemy flask-login
$ export FLASK_APP=project
$ export FLASK_DEBUG=1
```

We will not open up a python shell and create the database locally
```bash
$ python3
```

Run these commands in python one time to create the local DB
```python
from project import db, create_app
db.create_all(app=create_app()) 
# pass the create_app result so Flask-SQLAlchemy gets the configuration.
```
You should see a db.sqlite file

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


## Credit
1. Thank you to digitalocean for [tutorial on flask auth](https://www.digitalocean.com/community/tutorials/how-to-add-authentication-to-your-app-with-flask-login)
2. [SamuelSacco](https://github.com/SamuelSacco) for creating the frontend