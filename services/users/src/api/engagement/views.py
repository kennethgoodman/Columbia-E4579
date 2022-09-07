from flask import request
from flask_restx import Namespace, Resource, fields

from src.api.engagement.crud import (  # isort:skip
    get_engagement_by_id,
    get_all_engagements_by_content_id,
    get_all_engagements_by_user_id,
    get_engagement_by_content_and_user_and_type,
    add_engagement,
    increment_engagement,
    delete_engagement
)
from src.api.engagement.models import (
    EngagementType
)

engagement_namespace = Namespace("engagement")


engagement = engagement_namespace.model(
    "Engagement",
    {
        "id": fields.Integer(readOnly=True),
        "content_id": fields.Integer(required=True),
        "engagement_type": fields.String(description='engagement type', enum=EngagementType._member_names_, required=True),
        "engagement_value": fields.Integer(required=False),
        "created_date": fields.DateTime,
    },
)


class EngagementList(Resource):
    @engagement_namespace.marshal_with(engagement)
    def get(self, engagement_id):
        """Returns all users."""
        return get_engagement_by_id(engagement_id), 200


class EngagementManager(Resource):
    @engagement_namespace.expect(engagement, validate=True)
    @engagement_namespace.response(400, "Engagement already exists")
    def post(self):
        """Creates a new engagement."""
        post_data = request.get_json()
        user_id = post_data.get("user_id")
        content_id = post_data.get("content_id")
        engagement_type = post_data.get("engagement_type")
        engagement_value = post_data.get("engagement_value")
        response_object = {}

        engagement = get_engagement_by_content_and_user_and_type(user_id, content_id, engagement_type)
        if engagement:
            response_object["message"] = "Engagement already exists"
            return response_object, 400

        add_engagement(user_id, content_id, engagement_type, engagement_value)
        response_object["message"] = f"engagement was added!"
        return response_object, 200

engagement_namespace.add_resource(EngagementManager)
engagement_namespace.add_resource(EngagementList, "/<int:engagement_id>")
