from .AbstractModel import AbstractModel
from project.ml_models.untrained_model.not_training import ModelController
import tensorflow as tf
import numpy as np

# load model once
model = ModelController('untrained', load_model=True).model


class UntrainedModel(AbstractModel):
    def _create_idv_data(self, content_id, user_id):
        return np.array([content_id, user_id])

    def _create_all_data(self, content_ids, user_id):
        return np.array(
            list(
                map(
                    lambda content_id: self._create_idv_data(content_id, user_id),
                    content_ids
                )
            )
        ).reshape((len(content_ids), 2))

    def predict_probabilities(self, content_ids, user_id):
        predictions = model(self._create_all_data(content_ids, user_id)).numpy()
        predictions = tf.nn.softmax(predictions).numpy()
        return list(map(lambda i: {
            "content_id": content_ids[i],
            "p_engage": predictions[i][0],  # hard coding that first output is p(Engage | data)
        }, range(len(content_ids))))
