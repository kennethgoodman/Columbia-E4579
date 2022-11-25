from enum import Enum

from src.api.content.models import Content, get_url
from src.api.users.models import User
from src.recommendation_system.recommendation_flow.controllers import (
    RandomController,
    ExampleController,
    AlphaController,
    BetaController,
    CharlieController,
    DeltaController
)


class ControllerEnum(Enum):
    RANDOM = RandomController
    EXAMPLE = ExampleController
    ALPHA = AlphaController
    BETA = BetaController
    CHARLIE = CharlieController
    DELTA = DeltaController

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


def get_content_data(controller, user_id, limit, offset, seed, starting_point=None):
    if controller in [
        ControllerEnum.RANDOM,
        ControllerEnum.EXAMPLE,
        ControllerEnum.ALPHA,
        ControllerEnum.BETA,
        ControllerEnum.CHARLIE,
        ControllerEnum.DELTA
    ]:
        content_ids = controller.value().get_content_ids(
            user_id, limit, offset, seed, starting_point
        )
    else:
        raise ValueError(f"don't support that controller: {controller}")
    all_content = Content.query.filter(Content.id.in_(content_ids)).all()
    responses = map(content_to_response, all_content)
    return list(responses)
