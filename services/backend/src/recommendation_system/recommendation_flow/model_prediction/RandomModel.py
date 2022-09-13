import random

from .AbstractModel import AbstractModel


class RandomModel(AbstractModel):
    def predict_probabilities(self, content_ids, user_id, seed=None):
        if seed:
            random.seed(seed)
        return list(
            map(
                lambda content_id: {
                    "content_id": content_id,
                    "p_engage": random.random(),
                },
                content_ids,
            )
        )
