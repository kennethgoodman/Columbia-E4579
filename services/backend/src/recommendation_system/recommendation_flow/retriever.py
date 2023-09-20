from enum import Enum
import time

from src.api.content.models import Content, get_url
from src.api.users.models import User
from src.recommendation_system.recommendation_flow.controllers import (
    RandomController,
    ExampleController,
)

from src.api.metrics.models import MetricFunnelType, MetricType, TeamName
from src.api.metrics.crud import add_metric


class ControllerEnum(Enum):
    RANDOM = RandomController
    EXAMPLE = ExampleController

    def human_string(self):
        return str(self).split(".")[1]

    @staticmethod
    def string_to_controller(controller_string):
        return {
            controller.human_string(): controller for controller in list(ControllerEnum)
        }[controller_string]


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
        team_name=team_name, 
        funnel_name='retriever', 
        user_id=user_id if user_id else None, 
        content_id=None, 
        metric_funnel_type=MetricFunnelType.Controller, 
        metric_type=MetricType.TimeTaken, 
        metric_value=val,
        metric_metadata={
            "limit": limit, "offset": offset, 
            "seed": seed, "starting_point": starting_point
            }
    )


def get_content_data(controller, user_id, limit, offset, seed, starting_point=None):
    if controller in [
        ControllerEnum.RANDOM,
        ControllerEnum.EXAMPLE,
    ]:
        start = time.time()
        content_ids = controller.value().get_content_ids(
            user_id, limit, offset, seed, starting_point
        )
        try:
            add_metric_time_took({
                ControllerEnum.RANDOM: TeamName.Random,
                ControllerEnum.EXAMPLE: TeamName.Example
            }[controller], user_id, int(time.time() - start), 
                                limit, offset, seed, starting_point)
        except Exception as e:
            print(f"exception trying to add_metric_time_took {e}")
    else:
        raise ValueError(f"don't support that controller: {controller}")
    all_content = Content.query.filter(Content.id.in_(content_ids)).all()
    responses = map(content_to_response, all_content)
    return list(responses)
