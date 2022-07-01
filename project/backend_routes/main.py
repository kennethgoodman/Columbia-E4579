# main.py

from flask import Blueprint, render_template
from flask_login import login_required, current_user

main = Blueprint('main', __name__, static_folder='../frontend/build')

@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', username=current_user.username)


# Serve React App
@main.route('/')
def index():
    return main.send_static_file('index.html')