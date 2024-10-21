import operator
from sqlalchemy.sql.expression import func
from src import db
from src.api.content.models import Content, MediaType
from src.api.engagement.models import Engagement, EngagementType

from .AbstractGenerator import AbstractGenerator


class ExampleGenerator(AbstractGenerator):
    def _get_content_ids(self, _, limit, offset, _seed, starting_point):
        # TODO: should discount by creation_time so closer events have more weight
        results = (
            Engagement.query.with_entities(
                Engagement.content_id, func.count()
            )
            .filter_by(
                engagement_type=EngagementType.Like,
            )
            .group_by(Engagement.content_id)
            .order_by(func.count().desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        return list(map(lambda x: x[0], results)), list(map(lambda x: x[1], results))
    
    def _get_name(self):
        return "Example"


class ExampleGeneratorTextPct(AbstractGenerator):
    def _get_content_ids(self, _, limit, offset, _seed, starting_point):
        results_1 = (
            Engagement.query.with_entities(Engagement.content_id, func.count())
                .join(Content)
                .filter(Content.media_type == MediaType.Text)
                .filter(Engagement.engagement_type == EngagementType.Like)
                .group_by(Engagement.content_id)
                .order_by(func.count().desc())
                .limit(int(limit * 0.8)) 
                .offset(offset)
                .all()
        )
        results_2 = (
            Engagement.query.with_entities(Engagement.content_id, func.count())
                .join(Content)
                .filter(Content.media_type == MediaType.Image)
                .filter(Engagement.engagement_type == EngagementType.Like)
                .group_by(Engagement.content_id)
                .order_by(func.count().desc())
                .limit(int(limit * 0.2)) 
                .offset(offset)
                .all()
        )
        return list(map(lambda x: x[0], results_1)) + list(map(lambda x: x[0], results_2)), list(map(lambda x: x[1], results_1)) + list(map(lambda x: x[1], results_2))
    
    def _get_name(self):
        return "ExampleWithTextPct"
