from flask import request

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
                user_id=user_id if user_id else None, 
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
            print(f"exception trying to add_metric {team_name}, {self._get_name()}, {e}")
        return response

    def _get_content_ids(self, user_id, limit, offset, seed, starting_point):
        pass

    def _get_name(self):
        pass
