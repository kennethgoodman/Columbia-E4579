import os
import random

import jwt
from flask import jsonify, request
from flask_restx import Namespace, Resource, fields
from src.api.content.models import MediaType
from src.api.engagement.crud import (
    get_dislike_count_by_content_id,
    get_engagement_by_content_and_user_and_type,
    get_like_count_by_content_id,
)
from src.api.engagement.models import EngagementType, LikeDislike
from src.api.utils.auth_utils import get_user
from src.recommendation_system.recommendation_flow.controllers.RandomController import (
    RandomController,
)
from src.recommendation_system.recommendation_flow.retriever import (
    ControllerEnum,
    get_content_data,
)

content_namespace = Namespace("content")

parser = content_namespace.parser()
parser.add_argument("Authorization", location="headers")

controllers = content_namespace.model(
    "Controllers", {"controller": fields.String(required=True)}
)

content = content_namespace.model(
    "Content",
    {
        "id": fields.Integer(readOnly=True),
        "total_likes": fields.Integer(required=False),
        "total_dislikes": fields.Integer(required=False),
        "user_likes": fields.Boolean(required=False),
        "user_dislikes": fields.Boolean(required=False),
        "text": fields.String(required=False),
        "original_prompt": fields.String(required=False),
        "style": fields.String(required=False),
        "prompt": fields.String(requied=False),
        "author": fields.String(required=False),
        "width": fields.Integer(required=False),
        "height": fields.Integer(required=False),
        "url": fields.String(required=False),
        "download_url": fields.String(required=False),
    },
)


def add_content_data(responses, user_id):
    # TODO, can we do this all in one query to be faster?
    for response in responses:
        response["total_likes"] = get_like_count_by_content_id(response["id"])
        response["total_dislikes"] = get_dislike_count_by_content_id(response["id"])

        user_likes = get_engagement_by_content_and_user_and_type(
            user_id, response["id"], EngagementType.Like
        )
        response["user_likes"] = (
            user_likes.engagement_value == int(LikeDislike.Like)
            if user_likes
            else False
        )
        response["user_dislikes"] = (
            user_likes.engagement_value == int(LikeDislike.Dislike)
            if user_likes
            else False
        )

        if response.get("text") is None:
            response["text"] = response["author"]
    return responses


class ListControllers(Resource):
    @content_namespace.marshal_with(controllers, as_list=True)
    @content_namespace.response(200, "Success")
    def get(self):
        """Return a list of all possible controllers in their human-readable string format"""
        return [
            {"controller": controller.human_string()}
            for controller in list(ControllerEnum)
        ], 200


class ContentPagination(Resource):
    @content_namespace.marshal_with(content, as_list=True)
    @content_namespace.response(200, "Success")
    def get(self):
        """
        This API should be used for content pagination. Do not NEED to be signed in, but better experience if signed in
        """
        status_code, user_id, exception_message = get_user(request)
        if exception_message:
            print(exception_message)
            user_id = 0  # if error, do a logged out user, not great, TODO: ensure this is right
        page = int(request.args.get("page", 0))
        limit = int(request.args.get("limit", 10))
        content_id = request.args.get("content_id", None)
        if content_id == "undefined":
            content_id = None
        controller = ControllerEnum.string_to_controller(
            request.args.get("controller", ControllerEnum.RANDOM.human_string())
            or ControllerEnum.RANDOM.human_string()
        )
        seed = float(request.args.get("seed", random.random()))
        offset = page * limit
        # logged-out user is 0
        # don't need page for random (most of the time)
        starting_point = None
        if content_id is not None:
            starting_point = {"content_id": int(content_id)}
        responses = get_content_data(
            controller=controller,
            user_id=user_id,
            limit=limit,
            offset=offset,
            seed=seed,
            starting_point=starting_point,
        )
        return add_content_data(responses, user_id), 200


content_namespace.add_resource(ContentPagination, "")
content_namespace.add_resource(ContentPagination, "/similarcontent/<int:content_id>")
content_namespace.add_resource(ListControllers, "/listcontrollers")
