from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import current_user, login_required
from project.recommendation_flow.retriever import get_content_data, ControllerEnum

data_api = Blueprint('data_api', __name__, static_folder='../frontend/build', static_url_path='/')


@data_api.route('/api/get_images', methods=['GET'])
def get_images():
    urls = get_content_data(ControllerEnum.RANDOM, 0)  # logged out user is 0
    return {
        "images": [urls],
    }


@data_api.route('/api/joke', methods=['GET'])
def joke():
    return {
        "setup": "hi",
        "delivery": "hello"
    }


@data_api.route('/random_photos')
@login_required
def random_photo():
    urls = get_content_data(ControllerEnum.RANDOM, current_user.id)
    url_html = lambda idx_and_url: f"""<h1> Image {idx_and_url[0]} </h1> <img src="{idx_and_url[1]}">"""
    html = '<br>\n'.join(list(map(url_html, enumerate(urls))))
    return """
        <!DOCTYPE html>
        <html>
        <body>

        <h1>The img element</h1>
    """ + html + """
        </body>
        </html>
    """


@data_api.route('/ping', methods=['GET'])
def ping():
    return 'pong'
