import os
import jwt
from flask import request, jsonify
from flask_restx import Namespace, Resource, fields

from src.api.content.models import (
    MediaType
)
from src.api.engagement.crud import (
    get_like_count_by_content_id,
    get_dislike_count_by_content_id,
    get_engagement_by_content_and_user_and_type,
)
from src.api.utils.auth_utils import get_user
from src.api.engagement.models import (
    EngagementType, LikeDislike
)

content_namespace = Namespace("content")

parser = content_namespace.parser()
parser.add_argument("Authorization", location="headers")

content = content_namespace.model(
    "Content",
    {
        "id": fields.Integer(readOnly=True),
        'total_likes': fields.Integer(required=False),
        'total_dislikes': fields.Integer(required=False),
        'user_likes': fields.Boolean(required=False),
        'user_dislikes': fields.Boolean(required=False),
        'text': fields.String(required=False),
        'author': fields.String(required=False),
        'width': fields.Integer(required=False),
        'height': fields.Integer(required=False),
        'url': fields.String(required=False),
        'download_url': fields.String(required=False),
    },
)


def add_content_data(responses, user_id):
    # TODO, can we do this all in one query to be faster?
    for response in responses:
        response['total_likes'] = get_like_count_by_content_id(response['id'])
        response['total_dislikes'] = get_dislike_count_by_content_id(response['id'])

        user_likes = get_engagement_by_content_and_user_and_type(user_id, response['id'], EngagementType.Like)
        response['user_likes'] = user_likes.engagement_value == int(LikeDislike.Like) if user_likes else False
        response['user_dislikes'] = user_likes.engagement_value == int(LikeDislike.Dislike) if user_likes else False

        if response.get('text') is None:
            response['text'] = response['author']
    return responses


class ContentPagination(Resource):
    @content_namespace.marshal_with(content, as_list=True)
    def get(self):
        """Returns all content"""
        status_code, user_id, exception_message = get_user(request)
        if exception_message:
            print(exception_message)
            user_id = 0  # if error, do a logged out user, not great, TODO: ensure this is right
        page = int(request.args.get('page', 0))
        limit = int(request.args.get('limit', 10))
        offset = page * limit
        if int(os.environ.get("USE_PICSUM", "0")) == 1:
            import requests
            response = requests.get(f'https://picsum.photos/v2/list?page={page}&limit={limit}')
            return add_content_data(response.json(), user_id), 200
        # logged-out user is 0
        # don't need page for random (most of the time)
        responses = [] # get_content_data(controller=ControllerEnum.RANDOM, user_id=0, limit=max(limit, 50), offset=offset)
        return jsonify(add_content_data(responses, user_id)), 200


content_namespace.add_resource(ContentPagination, "")
