from flask import request
from flask_restx import Namespace, Resource, fields
from src.api.engagement.models import EngagementType, LikeDislike
from src.api.utils.auth_utils import get_user

from src.api.engagement.crud import (  # isort:skip
    get_all_engagements_by_content_id,
    get_engagement_by_content_and_user_and_type,
    get_engagement_count_by_content_id,
    add_engagement,
    update_engagement,
    delete_engagement,
)

engagement_namespace = Namespace("engagement")

engagement = engagement_namespace.model(
    "Engagement",
    {
        "id": fields.Integer(readOnly=True),
        "content_id": fields.Integer(required=True),
        "engagement_type": fields.String(
            description="engagement type",
            enum=EngagementType._member_names_,
            required=True,
        ),
        "engagement_value": fields.Integer(required=False),
        "created_date": fields.DateTime,
    },
)

set_engagement = engagement_namespace.model(
    "SetEngagement", {"message": fields.String(required=True)}
)

like_count = engagement_namespace.model(
    "LikeCount",
    {
        "count": fields.Integer(required=True),
        "content_id": fields.Integer(required=True),
    },
)


parser = engagement_namespace.parser()
parser.add_argument("Authorization", location="headers")


def post_like(user_id, content_id, likedislike):
    engagement_type = EngagementType.Like
    engagement = get_engagement_by_content_and_user_and_type(
        user_id, content_id, engagement_type
    )
    response_object = {}
    if engagement:
        if engagement.engagement_value == int(likedislike):
            response_object["message"] = "engagement already exists"
            return response_object, 400
        update_engagement(engagement, int(likedislike))
    else:
        add_engagement(user_id, content_id, engagement_type, int(likedislike))
    response_object["message"] = f"Success"
    return response_object, 200


def post_remove_likedislike(user_id, content_id, likedislike):
    engagement_type = EngagementType.Like
    engagement = get_engagement_by_content_and_user_and_type(
        user_id, content_id, engagement_type
    )
    response_object = {}
    if not engagement or engagement.engagement_value != int(likedislike):
        # if there isn't an engagement then we should allow undoing
        # if there is an engagement, make sure it's the right one
        response_object[
            "message"
        ] = "engagement doesn't exist or wrong engagement value"
        return response_object, 400
    delete_engagement(engagement)
    response_object["message"] = f"Success"
    return response_object, 200


def _get_all_engagements_by_content_id(content_id, engagement_type):
    response_object = {}
    count = get_all_engagements_by_content_id(content_id, EngagementType.Like)
    return {"count": count}, 200


class LikeCount(Resource):
    @engagement_namespace.marshal_with(like_count)
    def get(self, content_id):
        return get_engagement_count_by_content_id(content_id, EngagementType.Like)


class Like(Resource):
    @engagement_namespace.marshal_with(engagement)
    def get(self, content_id):
        """Retrieve the like count for this content_id"""
        return _get_all_engagements_by_content_id(content_id, EngagementType.Like)

    @engagement_namespace.expect(parser)
    @engagement_namespace.marshal_with(set_engagement)
    @engagement_namespace.response(200, "Success")
    @engagement_namespace.response(403, "Need To Login")
    @engagement_namespace.response(401, "Bad Token, need to re-login")
    @engagement_namespace.response(400, "Already exists")
    def post(self, content_id):
        """Like this content"""
        status_code, user_id, exception_message = get_user(request)
        if exception_message:
            engagement_namespace.abort(status_code, exception_message)
            return status_code, exception_message
        return post_like(user_id, content_id, LikeDislike.Like)


class Dislike(Resource):
    @engagement_namespace.marshal_with(engagement)
    def get(self, content_id):
        """Retrieve the dislike count for this content_id"""
        return _get_all_engagements_by_content_id(content_id, EngagementType.Like)

    @engagement_namespace.expect(parser)
    @engagement_namespace.marshal_with(set_engagement)
    @engagement_namespace.response(200, "Success")
    @engagement_namespace.response(403, "Need To Login")
    @engagement_namespace.response(401, "Bad Token, need to re-login")
    @engagement_namespace.response(400, "Already exists")
    def post(self, content_id):
        """Dislike this content"""
        status_code, user_id, exception_message = get_user(request)
        if exception_message:
            engagement_namespace.abort(status_code, exception_message)
            return status_code, exception_message
        return post_like(user_id, content_id, LikeDislike.Dislike)


class UnLike(Resource):
    @engagement_namespace.expect(parser)
    @engagement_namespace.marshal_with(set_engagement)
    @engagement_namespace.response(200, "Success")
    @engagement_namespace.response(403, "Need To Login")
    @engagement_namespace.response(401, "Bad Token, need to re-login")
    @engagement_namespace.response(400, "Already exists")
    def post(self, content_id):
        """Unlike this content"""
        status_code, user_id, exception_message = get_user(request)
        if exception_message:
            engagement_namespace.abort(status_code, exception_message)
            return status_code, exception_message
        return post_remove_likedislike(user_id, content_id, LikeDislike.Like)


class UnDislike(Resource):
    @engagement_namespace.expect(parser)
    @engagement_namespace.marshal_with(set_engagement)
    @engagement_namespace.response(200, "Success")
    @engagement_namespace.response(403, "Need To Login")
    @engagement_namespace.response(401, "Bad Token, need to re-login")
    @engagement_namespace.response(400, "Already exists")
    def post(self, content_id):
        """UnDislike this content"""
        status_code, user_id, exception_message = get_user(request)
        if exception_message:
            engagement_namespace.abort(status_code, exception_message)
            return status_code, exception_message
        return post_remove_likedislike(user_id, content_id, LikeDislike.Dislike)


class ElapsedTime(Resource):
    @engagement_namespace.expect(parser)
    @engagement_namespace.marshal_with(set_engagement)
    @engagement_namespace.response(200, "Success")
    @engagement_namespace.response(403, "Need To Login")
    @engagement_namespace.response(401, "Bad Token, need to re-login")
    @engagement_namespace.response(400, "Already exists")
    def post(self, content_id):
        status_code, user_id, exception_message = get_user(request)
        if exception_message:
            engagement_namespace.abort(status_code, exception_message)
            return status_code, exception_message
        add_engagement(
            user_id,
            content_id,
            EngagementType.MillisecondsEngagedWith,
            request.json["elapsed_time"],
        )
        return {"message": "Success"}, 200


engagement_namespace.add_resource(Like, "/like/<int:content_id>")
engagement_namespace.add_resource(UnLike, "/unlike/<int:content_id>")
engagement_namespace.add_resource(Dislike, "/dislike/<int:content_id>")
engagement_namespace.add_resource(UnDislike, "/undislike/<int:content_id>")
engagement_namespace.add_resource(LikeCount, "/likecount/<int:content_id>")
engagement_namespace.add_resource(ElapsedTime, "/elapsed_time/<int:content_id>")
