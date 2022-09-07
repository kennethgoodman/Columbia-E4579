import os
from flask import request, jsonify
from flask_restx import Namespace, Resource, fields

from src.api.content.models import (
    MediaType
)

content_namespace = Namespace("content")

parser = content_namespace.parser()
parser.add_argument("Authorization", location="headers")

content = content_namespace.model(
    "Content",
    {
        "id": fields.Integer(readOnly=True),
        'total_likes': fields.Integer(required=False),
        'user_likes': fields.Integer(required=False),
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
        total_likes, user_likes = 0, 0 # get_likes(response['id'], user_id)
        response['total_likes'] = total_likes
        response['user_likes'] = user_likes
        if response.get('text') is None:
            response['text'] = response['author']
    return responses


class ContentPagination(Resource):
    @content_namespace.marshal_with(content, as_list=True)
    def get(self):
        """Returns all content"""
        auth_header = request.headers.get("Authorization")
        if auth_header:
            try:
                access_token = auth_header.split(" ")[1]
                user_id = User.decode_token(access_token)
            except jwt.ExpiredSignatureError:
                content_namespace.abort(401, "Signature expired. Please log in again.")
                return "Signature expired. Please log in again.", 401
            except jwt.InvalidTokenError:
                content_namespace.abort(401, "Invalid token. Please log in again.")
                return "Invalid token. Please log in again.", 401
        else:
            user_id = 0  # logged out user
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
