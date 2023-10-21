from src import db
from flask import request
import traceback

from src.api.metrics.models import MetricFunnelType, MetricType
from src.api.metrics.crud import add_metric


class AbstractGenerator:
    def get_content_ids(self, team_name, user_id, limit, offset, seed, starting_point):
        response = self._get_content_ids(user_id, limit, offset, seed, starting_point)
        try:
            add_metric(
                request_id=request.request_id,
                team_name=team_name,
                funnel_name=self._get_name(),
                user_id=user_id if user_id not in [None, 0] else None, 
                content_id=None,
                metric_funnel_type=MetricFunnelType.CandidateGeneration,
                metric_type=MetricType.CandidateGenerationNumCandidates,
                metric_value=len(response[0]) if response is not None and len(response) == 2 else -1,
                metric_metadata={
                    "limit": limit, "offset": offset,
                    "seed": seed, "starting_point": starting_point
                    }
            )
        except Exception as e:
            db.session.rollback()
            print(f"exception trying to add_metric {team_name}, {user_id}, {self._get_name()}, {e}")
            print(traceback.format_exc())
        return response

    def _get_content_ids(self, user_id, limit, offset, seed, starting_point):
        raise NotImplementedError("you need to implement")

    def _get_name(self):
        raise NotImplementedError("you need to implement")
