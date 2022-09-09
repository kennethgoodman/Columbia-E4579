from .AbstractModel import AbstractModel
import random


class RandomModel(AbstractModel):
    def predict_probabilities(self, content_ids, user_id):
        return list(map(lambda content_id: {
            "content_id": content_id,
            "p_engage": random.random(),
        }, content_ids))
