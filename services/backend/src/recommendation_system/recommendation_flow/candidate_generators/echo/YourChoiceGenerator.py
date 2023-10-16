from sqlalchemy.sql.expression import func, text
from src import db
from src.api.engagement.models import Engagement, EngagementType
from src.recommendation_system.recommendation_flow.candidate_generators.AbstractGenerator import AbstractGenerator

class YourChoiceGenerator(AbstractGenerator):
    def _get_content_ids(self, _, limit, offset, _seed, starting_point):
        results = (
            Engagement.query.with_entities(
                Engagement.content_id, func.sum((Engagement.engagement_value) * (315360000 - func.timestampdiff(text("SECOND"), Engagement.created_date, func.now()))).label('Weighted Likeability')
            )
            .filter_by(
                engagement_type=EngagementType.Like,
            )
            .group_by(Engagement.content_id)
            .order_by(func.sum((Engagement.engagement_value) * (315360000 - func.timestampdiff(text("SECOND"), Engagement.created_date, func.now()))).desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        return list(map(lambda x: x[0], results))[:725], ([0]*len(results))[:725]
    
    def _get_name(self):
        return "YourChoiceGenerator"
