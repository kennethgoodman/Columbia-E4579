from flask import Blueprint, request, jsonify
from flask_login import current_user, login_required
import sqlalchemy.exc
from project.data_models.engagement import Engagement, EngagementType
from project import db
from project.query_utils.engagement import get_likes

engagement_api = Blueprint('engagement_api', __name__, static_folder='../frontend/build', static_url_path='/')


@engagement_api.route('/api/engagement/like_content', methods=['POST'])
@login_required
def like_content():
    content_id = request.args.get('content_id')
    # create new user with the form data. Hash the password so plaintext version isn't saved.
    new_engagement = Engagement(user_id=current_user.id, content_id=content_id, engagement_type=EngagementType.Like)
    # add the new user to the database
    db.session.add(new_engagement)
    try:
        db.session.commit()
    except sqlalchemy.exc.IntegrityError:
        pass  # already liked
    return jsonify({'success': True})


@engagement_api.route('/api/engagement/unlike_content', methods=['POST'])
@login_required
def unlike_content():
    content_id = request.args.get('content_id')
    Engagement.query\
        .filter_by(user_id=current_user.id, content_id=content_id, engagement_type=EngagementType.Like)\
        .delete()
    db.session.commit()
    return jsonify({'success': True})


@engagement_api.route('/api/engagement/total_likes', methods=['GET'])
def total_likes():
    content_id = request.args.get('content_id')
    return jsonify({'success': True, 'total_likes': get_likes(content_id)[0]})
