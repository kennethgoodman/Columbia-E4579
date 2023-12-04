import random

from .AbstractModel import AbstractModel


class RandomModel(AbstractModel):
    def _predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):
        if seed:
            random.seed(seed)
        content_ids = list(content_ids)
        like = [random.random() for _ in content_ids]
        dislike = [random.random() for _ in content_ids]
        eng = [random.random() for _ in content_ids]
        return like, dislike, eng, content_ids

    def _get_name(self):
        return "RandomModel"

