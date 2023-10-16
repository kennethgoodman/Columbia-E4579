from enum import Enum
import time
import random
from src import db
from flask import request

from src.api.content.models import Content, get_url
from src.api.users.models import User
from src.recommendation_system.recommendation_flow.controllers import (
    RandomController,
    ExampleController,
    EngagementTimeController,
    StaticController,
    AlphaController,
    BetaController,
    CharlieController,
    DeltaController,
    EchoController,
    FoxtrotController,
    GolfController
)

from src.api.metrics.models import MetricFunnelType, MetricType, TeamName
from src.api.metrics.crud import add_metric


class ControllerEnum(Enum):
    RANDOM = RandomController
    EXAMPLE = ExampleController
    ENGAGEMENT_TIME = EngagementTimeController
    STATIC = StaticController
    ALPHA = AlphaController
    BETA = BetaController
    CHARLIE = CharlieController
    DELTA = DeltaController
    ECHO = EchoController
    FOXTROT = FoxtrotController
    GOLF = GolfController

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
            ControllerEnum.ALPHA: TeamName.Alpha_F2023,
            ControllerEnum.BETA: TeamName.Beta_F2023,
            ControllerEnum.CHARLIE: TeamName.Charlie_F2023,
            ControllerEnum.DELTA: TeamName.Delta_F2023,
            ControllerEnum.ECHO: TeamName.Echo_F2023,
            ControllerEnum.FOXTROT: TeamName.Foxtrot_F2023,
            ControllerEnum.GOLF: TeamName.Golf_F2023,
            ControllerEnum.ENGAGEMENT_TIME: TeamName.EngagementTime,
            ControllerEnum.STATIC: TeamName.Static
        }[controller]


def content_to_response(content):
    generated_content_metadata = content.generated_content_metadata
    return {
        "id": content.id,
        "download_url": get_url(content),
        "author": generated_content_metadata.source,  # TODO: change to a query?
        "text": f"""{generated_content_metadata.original_prompt}\n In the style of {generated_content_metadata.artist_style}""",
        "prompt": generated_content_metadata.prompt,
        "style": generated_content_metadata.artist_style,
        "original_prompt": generated_content_metadata.original_prompt,
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


def get_content_data(controller, user_id, limit, offset, seed, starting_point=None):
    start = time.time()
    try:
        if False: # controller == ControllerEnum.ENGAGEMENT_ASSIGNMENT:
            content_ids, new_controller = controller.value().get_content_ids(
                user_id, limit, offset, seed, starting_point
            )
            controller = ControllerEnum.controller_to_string(new_controller)
        else:
            content_ids = controller.value().get_content_ids(
                user_id, limit, offset, seed, starting_point
            )
    except Exception as e:
        print(f"exception trying to get content for {controller}, {e}")
        content_ids = []
    try:
        add_metric_time_took(ControllerEnum.controller_to_team_name(controller), 
                             user_id, int(1000 * (time.time() - start)), 
                             limit, offset, seed, starting_point)
    except Exception as e:
        db.session.rollback()
        print(f"exception trying to add_metric_time_took {e}")
    all_content = Content.query.filter(Content.id.in_(content_ids)).all()
    responses = map(content_to_response, all_content)
    return list(map(lambda x: {**x, "controller": controller.human_string()}, responses))
