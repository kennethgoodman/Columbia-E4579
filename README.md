# Columbia-E4579
Fall 2022 Class At Columbia. Modern Recommendation Systems

## Dev Setup

```bash
$ python3 -m venv E4579
$ source E4579/bin/activate
$ pip install flask flask-sqlalchemy flask-login
$ export FLASK_APP=project
$ export FLASK_DEBUG=1
```

```bash
$ python3
```

```python
from project import db, create_app
db.create_all(app=create_app()) 
# pass the create_app result so Flask-SQLAlchemy gets the configuration.
```

```bash
$ flask run
```

If you get an error that flask-sqlalchemy or flask-login doesn't exist, you have two options:
1. uninstall flask locally, quit your local venv by running `deactivate` or opening a new shell
```bash
pip uninstall flask
```
2. install flask-sqlalchemy and flask-login globally, quit your local venv by running `deactivate` or opening a new shell
```bash
pip install flask-sqlalchemy flask-login
```