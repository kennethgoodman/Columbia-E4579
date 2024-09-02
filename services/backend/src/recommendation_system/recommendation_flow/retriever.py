from enum import Enum
import time
import random
from src import db
from flask import request
import traceback

from src.api.content.models import Content, get_url
from src.api.users.models import User
from src.recommendation_system.recommendation_flow.controllers import (
    RandomController,
    ExampleController,
    EngagementTimeController,
    StaticController
)

from src.api.metrics.models import MetricFunnelType, MetricType, TeamName
from src.api.metrics.crud import add_metric


class ControllerEnum(Enum):
    RANDOM = RandomController
    EXAMPLE = ExampleController
    ENGAGEMENT_TIME = EngagementTimeController
    STATIC = StaticController

    def human_string(self):
        return str(self).split(".")[1]

    @staticmethod
    def string_to_controller(controller_string):
        return {
            controller.human_string(): controller for controller in list(ControllerEnum)
        }[controller_string]

    @staticmethod
    def controller_to_string(controller):
        for controller_enum in list(ControllerEnum):
            if controller_enum.value() == controller:
                return controller_enum
        raise ValueError(f"{str(controller)} doesn't exist")

    @staticmethod
    def controller_to_team_name(controller):
        return {
            ControllerEnum.RANDOM: TeamName.Random,
            ControllerEnum.EXAMPLE: TeamName.Example,
            ControllerEnum.ENGAGEMENT_TIME: TeamName.EngagementTime,
            ControllerEnum.STATIC: TeamName.Static
        }[controller]


def content_to_response(content):
    generated_content_metadata = content.generated_content_metadata
    text = "N/A ERROR"
    url = get_url(content)
    if url is not None:
        text = f"""{generated_content_metadata.original_prompt}\n In the style of {generated_content_metadata.artist_style}"""
    else:
        text = generated_content_metadata.text
    return {
        "id": content.id,
        "download_url": get_url(content),
        "author": generated_content_metadata.source,  # TODO: change to a query?
        "text": text,
        "prompt": generated_content_metadata.prompt,
        "style": generated_content_metadata.artist_style,
        "original_prompt": generated_content_metadata.original_prompt,
        "type": "image" if url is not None else "text"
    }

def add_metric_time_took(team_name, user_id, val, limit, offset, seed, starting_point):
    add_metric(
        request_id=request.request_id,
        team_name=team_name,
        funnel_name='retriever',
        user_id=user_id if user_id else None,
        content_id=None,
        metric_funnel_type=MetricFunnelType.Controller,
        metric_type=MetricType.TimeTakenMS,
        metric_value=val,
        metric_metadata={
            "limit": limit, "offset": offset,
            "seed": seed, "starting_point": starting_point
            }
    )


def get_content_data(controller, user_id, limit, offset, seed, starting_point):
    start = time.time()

    content_ids = controller.value().get_content_ids(
        user_id, limit, offset, seed, starting_point
    )
    try:
        add_metric_time_took(ControllerEnum.controller_to_team_name(controller),
                             user_id, int(1000 * (time.time() - start)),
                             limit, offset, seed, starting_point)
    except Exception as e:
        db.session.rollback()
        print(f"exception trying to add_metric_time_took {e}")
        print(traceback.format_exc())
    all_content = Content.query.filter(Content.id.in_(content_ids)).all()
    responses = map(content_to_response, all_content)
    return list(map(lambda x: {**x, "controller": controller.human_string()}, responses))
