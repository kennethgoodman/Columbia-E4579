from flask import request
from flask_restx import Namespace, Resource, fields
from src.api.metrics.models import MetricType
from src.api.utils.auth_utils import get_user

from src.api.metric.crud import (  # isort:skip

)

metric_namespace = Namespace("metric")

metric = metric_namespace.model(
    "Metric",
    {
        "id": fields.Integer(readOnly=True),
        "user_id": fields.Integer(required=False),
        "content_id": fields.Integer(required=False),
        "metric_type": fields.String(
            description="metric type",
            enum=EngagementType._member_names_,
            required=True,
        ),
        "metric_value": fields.Integer(required=False),
        "created_date": fields.DateTime,
    },
)