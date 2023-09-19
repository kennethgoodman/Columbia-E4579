from sqlalchemy import func
from src import db
from src.api.metrics.models import Metric, MetricType

def get_all_metrics():
    return Metric.query.all()

def get_engagement_by_id(metric_id):
    return Metric.query.filter_by(id=metric_id).all()

def add_metric(team_name, funnel_name, user_id, content_id, 
    metric_funnel_type, metric_type, metric_value, metric_metadata):
    metric = Metric(
        team_name=team_name,
        funnel_name=funnel_name,
        user_id=user_id,
    	content_id=content_id,
        metric_funnel_type=metric_funnel_type,
        metric_type=metric_type,
    	metric_value=metric_value,
    	metric_metadata=metric_metadata
    )
    db.session.add(metric)
    db.session.commit()
    return metric