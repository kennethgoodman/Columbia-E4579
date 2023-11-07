from src import db
from flask import request
import traceback
import time
import random
from flask import current_app

from src.api.metrics.models import MetricFunnelType, MetricType
from src.api.metrics.crud import add_metric

class AbstractFilter:
    def filter_ids(self, team_name, user_id, content_ids, seed, starting_point):
        start = time.time()
        max_cans = current_app.config.get("MAX_CANDIDATES_TO_FILTERING")
        if len(content_ids) > max_cans:
            content_ids = random.sample(content_ids, max_cans)
        response = self._filter_ids(user_id, content_ids, seed, starting_point)
        end = time.time()
        try:
            add_metric(
                request_id=request.request_id,
                team_name=team_name,
                funnel_name=self._get_name(),
                user_id=user_id if user_id not in [None, 0] else None,
                content_id=None,
                metric_funnel_type=MetricFunnelType.Filtering,
                metric_type=MetricType.FilteringNumCandidates,
                metric_value=len(response) if response is not None else -1,
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
                metric_funnel_type=MetricFunnelType.Filtering,
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
        if starting_point.get('inverseFilter', False):
            # get the filtered out candidates
            return list(set(content_ids) - set(response))
        return response

    def _filter_ids(self, user_id, content_ids, seed, starting_point):
        raise NotImplementedError("you need to implement")

    def _get_name(self):
        raise NotImplementedError("you need to implement")


