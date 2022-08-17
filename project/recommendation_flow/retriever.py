from project.recommendation_flow.controllers.RandomController import RandomController
from project.data_models.content import Content, get_url
from enum import Enum


class ControllerEnum(Enum):
    RANDOM = RandomController


def content_to_response(content):
    return {
        'download_url': get_url(content),
        'author': content.author_id,
        'text': content.text
    }


def get_content_data(controller, user_id, limit):
    if controller == ControllerEnum.RANDOM:
        content_ids = ControllerEnum.RANDOM.value().get_content_ids(user_id, limit)
    else:
        raise ValueError(f"don't support that controller: {controller}")
    all_content = Content.query.filter(Content.id.in_(content_ids)).all()
    responses = map(content_to_response, all_content)
    return list(responses)
