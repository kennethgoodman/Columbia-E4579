from src import db
from flask import request
import traceback
import time
import random

from src.api.metrics.models import MetricFunnelType, MetricType
from src.api.metrics.crud import add_metric


class AbstractModel:
    def predict_probabilities(self, team_name, content_ids, user_id, seed=None, starting_point=None, **kwargs):
        start = time.time()
        response = self._predict_probabilities(
            content_ids, user_id, seed, **kwargs
        )
        end = time.time()
        try:
            for i, name in enumerate(["Like", "Dislike", "EngTime"]):
                add_metric(
                    request_id=request.request_id,
                    team_name=team_name,
                    funnel_name=self._get_name() + "_" +  name,
                    user_id=user_id if user_id not in [None, 0] else None,
                    content_id=None,
                    metric_funnel_type=MetricFunnelType.Prediction,
                    metric_type=MetricType.PredictionMeanPredicted,
                    metric_value=sum(response[i]) / len(response[i]) if len(response[i]) else 0,
                    metric_metadata={
                        "seed": seed, "starting_point": starting_point
                    }
                )
                # random sample 5 ids to write for prediction
                for response_idx in random.sample(range(len(response[i])), min(5, len(response[i]))):
                    add_metric(
                        request_id=request.request_id,
                        team_name=team_name,
                        funnel_name=self._get_name() + "_" + name,
                        user_id=user_id if user_id not in [None, 0] else None,
                        content_id=response[3][response_idx],
                        metric_funnel_type=MetricFunnelType.Prediction,
                        metric_type=MetricType.PredictionMeanPredicted,
                        metric_value=response[i][response_idx],
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
                metric_funnel_type=MetricFunnelType.Prediction,
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
        return response

    def _predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):
        raise NotImplementedError("you need to implement this")

    def _get_name(self):
        raise NotImplementedError("you need to implement this")
