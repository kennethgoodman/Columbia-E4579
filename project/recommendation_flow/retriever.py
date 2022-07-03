from project.recommendation_flow.controllers.RandomController import RandomController
from project.models.content import Content, get_url
from enum import Enum


class ControllerEnum(Enum):
    RANDOM = RandomController


def get_content_data(controller):
    if controller == ControllerEnum.RANDOM:
        content_ids = ControllerEnum.RANDOM.value().get_content_ids()
    else:
        raise ValueError(f"don't support that controller: {controller}")
    all_content = Content.query.filter(Content.id.in_(content_ids)).all()
    urls = map(get_url, all_content)
    return list(urls)
