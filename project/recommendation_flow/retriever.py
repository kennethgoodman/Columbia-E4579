from project.recommendation_flow.controllers.RandomController import RandomController
from project.data_models.content import Content, get_url
from enum import Enum


class ControllerEnum(Enum):
    RANDOM = RandomController


def get_content_data(controller, user_id):
    if controller == ControllerEnum.RANDOM:
        content_ids = ControllerEnum.RANDOM.value().get_content_ids(user_id)
    else:
        raise ValueError(f"don't support that controller: {controller}")
    all_content = Content.query.filter(Content.id.in_(content_ids)).all()
    urls = map(get_url, all_content)
    return list(urls)
