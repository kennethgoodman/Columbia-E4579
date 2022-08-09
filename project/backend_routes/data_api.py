from flask import Blueprint, request, current_app, jsonify
from flask_login import current_user, login_required
from project.recommendation_flow.retriever import get_content_data, ControllerEnum

data_api = Blueprint('data_api', __name__, static_folder='../frontend/build', static_url_path='/')


@data_api.route('/api/get_images', methods=['GET'])
def get_images():
    page = request.args.get('page')
    limit = request.args.get('limit')
    current_app.logger.info('in get images', page, limit)
    if current_app.config.get("use_picsum"):
        import requests
        response = requests.get(f'https://picsum.photos/v2/list?page={page}&limit={limit}')
        return jsonify(response.json())
    # logged out user is 0
    # don't need page for random (most of the time)
    responses = get_content_data(controller=ControllerEnum.RANDOM, user_id=0, limit=limit)
    return jsonify(responses)


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
