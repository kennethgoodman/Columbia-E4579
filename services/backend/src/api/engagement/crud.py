from sqlalchemy import func
from src import db
from src.api.engagement.models import Engagement, EngagementType, LikeDislike


def get_all_engagements():
    return Engagement.query.all()


def get_engagement_by_id(engagement_id):
    return Engagement.query.filter_by(id=engagement_id).all()


def _get_engagements_query_by_content_id(content_id, engagement_type=None):
    if engagement_type is None:
        return Engagement.query.filter_by(content_id=content_id)
    return Engagement.query.filter_by(
        content_id=content_id, engagement_type=engagement_type
    )


def get_all_engagements_by_content_id(content_id, engagement_type=None):
    return _get_engagements_query_by_content_id(content_id, engagement_type).all()


def get_engagement_count_by_content_id(content_id, engagement_type=None):
    return (
        _get_engagements_query_by_content_id(content_id, engagement_type)
        .with_entities(func.count())
        .scalar()
    )


def get_like_count_by_content_id(content_id):
    return (
        _get_engagements_query_by_content_id(content_id, EngagementType.Like)
        .filter_by(engagement_value=int(LikeDislike.Like))
        .with_entities(func.count())
        .scalar()
    )


def get_dislike_count_by_content_id(content_id):
    return (
        _get_engagements_query_by_content_id(content_id, EngagementType.Like)
        .filter_by(engagement_value=int(LikeDislike.Dislike))
        .with_entities(func.count())
        .scalar()
    )


def get_all_engagements_by_user_id(user_id):
    return Engagement.query.filter_by(user_id=user_id).all()


def get_engagement_by_content_and_user_and_type(user_id, content_id, engagement_type):
    return Engagement.query.filter_by(
        user_id=user_id, content_id=content_id, engagement_type=engagement_type
    ).first()


def add_engagement(user_id, content_id, engagement_type, engagement_value):
    if engagement_value is not None:
        engagement = Engagement(
            user_id=user_id,
            content_id=content_id,
            engagement_type=engagement_type,
            engagement_value=engagement_value,
        )
    else:
        engagement = Engagement(
            user_id=user_id, content_id=content_id, engagement_type=engagement_type
        )
    db.session.add(engagement)
    db.session.commit()
    return engagement


def update_engagement(engagement, engagement_value):
    engagement.engagement_value = engagement_value
    db.session.commit()


def increment_engagement(engagement_id, increment):
    engagement = (
        db.session.query(Engagement)
        .with_for_update()
        .filter_by(id=engagement_id)
        .first()
    )
    engagement.engagement_value += increment
    db.session.commit()
    return engagement


def delete_engagement(engagement):
    db.session.delete(engagement)
    db.session.commit()
    return
