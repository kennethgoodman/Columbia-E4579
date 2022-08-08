# main.py

from flask import Blueprint, render_template, current_app, request
from flask_login import login_required, current_user

main = Blueprint('main', __name__, static_folder='../frontend/build')


@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', username=current_user.username)


@main.route('/random_photo')
def random_photo():
    image_url = "https://columbia-fall2022-e4579-images.s3.amazonaws.com/Images/n02085620-Chihuahua/n02085620_10074.jpg"
    return f"""
        <!DOCTYPE html>
        <html>
        <body>
        
        <h1>The img element</h1>
        
        <img src="{image_url}">
        
        </body>
        </html>
    """


# Serve React App
@main.route("/app/", defaults={"path": "index.html"})
@main.route("/app/<path:path>")
def index(path):
    if current_app.config.get("FLASK_DEBUG") == 1:
        return current_app.proxy(request.path)
    return main.send_static_file(path)
