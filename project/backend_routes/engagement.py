from flask import Blueprint
from flask_login import current_user, login_required

engagement_api = Blueprint('engagement_api', __name__, static_folder='../frontend/build', static_url_path='/')


@engagement_api.route('/api/engagement/like_content')
@login_required
def like_content():
    return {'success': True}


@engagement_api.route('/api/engagement/like_content')
@login_required
def unlike_content():
    return {'success': True}


@engagement_api.route('/api/engagement/total_likes')
def total_likes():
    return {'success': True, 'total_likes': 0}