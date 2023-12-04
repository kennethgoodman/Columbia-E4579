from src import db
from flask import request
import traceback
import time

from src.api.metrics.models import MetricFunnelType, MetricType
from src.api.metrics.crud import add_metric


class AbstractRanker:
    def rank_ids(self, team_name, user_id, content_ids, limit, probabilities, seed, starting_point, X=None):
        start = time.time()
        response = self._rank_ids(
            user_id, content_ids, limit, probabilities, seed, starting_point, X
        )
        end = time.time()
        try:
            add_metric(
                request_id=request.request_id,
                team_name=team_name,
                funnel_name=self._get_name(),
                user_id=user_id if user_id not in [None, 0] else None,
                content_id=None,
                metric_funnel_type=MetricFunnelType.Ranking,
                metric_type=MetricType.RankingNumCandidates,
                metric_value=len(response),
                metric_metadata={
                    "seed": seed, "starting_point": starting_point
                }
            )
            add_metric(
                request_id=request.request_id,
                team_name=team_name,
                funnel_name=self._get_name(),
                user_id=user_id if user_id not in [None, 0] else None,
                content_id=None,
                metric_funnel_type=MetricFunnelType.Ranking,
                metric_type=MetricType.RankedCandidates,
                metric_value=10,  # first 10 candidates stored
                metric_metadata={
                    "seed": seed,
                    "starting_point": starting_point,
                    "candidates": ','.join(map(str, response[:10]))
                }
            )
            add_metric(
                request_id=request.request_id,
                team_name=team_name,
                funnel_name=self._get_name(),
                user_id=user_id if user_id not in [None, 0] else None,
                content_id=None,
                metric_funnel_type=MetricFunnelType.Ranking,
                metric_type=MetricType.TimeTakenMS,
                metric_value=int(1000 * (end - start)),
                metric_metadata={
                    "seed": seed, "starting_point": starting_point
                }
            )
        except Exception as e:
            db.session.rollback()
            print(f"exception trying to add_metric {team_name}, {user_id}, {self._get_name()}, {e}")
            print(traceback.format_exc())
        if starting_point.get('inverseRanker'):
            print("inverseing")
            response.reverse()  # inverse it
        return response

    def _rank_ids(self, user_id, content_ids, limit, probabilities, seed, starting_point, X=None):
        raise NotImplementedError("you need to implement this")

    def _get_name(self):
        raise NotImplementedError("you need to implement this")
