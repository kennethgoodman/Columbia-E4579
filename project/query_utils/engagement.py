from project.data_models.engagement import Engagement


def get_likes(content_id, user_id=None):
    """
    :param content_id: the content_id
    :param user_id: if user_id is None then only query for the content as if logged out
    :return: tuple of likes for the content and if the user has liked it
    """
    results = Engagement.query.filter_by(content_id=content_id).all()
    user_id_likes = None
    if user_id is not None:
        user_id_likes = Engagement.query.filter_by(content_id=content_id, user_id=user_id).first() is not None
    return len(results), user_id_likes
