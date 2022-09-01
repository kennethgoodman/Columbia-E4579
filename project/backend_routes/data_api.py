from flask import Blueprint, request, current_app, jsonify, session
from flask_login import current_user, login_required
from project.query_utils.engagement import get_likes
from project.recommendation_flow.retriever import get_content_data, ControllerEnum

data_api = Blueprint('data_api', __name__, static_folder='../frontend/build', static_url_path='/')


def add_content_data(responses, user_id):
    # TODO, can we do this all in one query to be faster?
    for response in responses:
        total_likes, user_likes = get_likes(response['id'], user_id)
        response['total_likes'] = total_likes
        response['user_likes'] = user_likes
        if response.get('text') is None:
            response['text'] = response['author']
    return responses


@data_api.route('/api/get_images', methods=['GET'])
def get_images():
    # TODO: ensure page and limit exist. Instead of page, use offset directly to allow different limits
    page = int(request.args.get('page', 0))
    limit = int(request.args.get('limit', 10))
    offset = page * limit
    if current_app.config.get("use_picsum"):
        import requests
        response = requests.get(f'https://picsum.photos/v2/list?page={page}&limit={limit}')
        return jsonify(add_content_data(response.json(), current_user.id))
    # logged-out user is 0
    # don't need page for random (most of the time)
    responses = get_content_data(controller=ControllerEnum.RANDOM, user_id=0, limit=limit, offset=offset)
    return jsonify(add_content_data(responses, current_user.id))


@data_api.route('/random_photos')
@login_required
def random_photo():
    urls = get_content_data(ControllerEnum.RANDOM, current_user.id, limit=10)
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
