import os

import tensorflow as tf


#  https://www.tensorflow.org/tutorials/quickstart/beginner
class ModelController:
    def __init__(self, model_name, load_model=False):
        self.model_name = model_name
        self.model = self.get_model()
        if load_model:
            latest = tf.train.latest_checkpoint(self._checkpoint_path)
            try:
                self.model.load_weights(latest)
            except:
                pass  # no model to load from

    def train_model(self):
        pass

    @property
    def _checkpoint_path(self):
        return os.path.join("model_weights", self.model_name)

    @property
    def checkpoint_filepath(self):
        return os.path.join(self._checkpoint_path, "cp-{epoch:04d}.ckpt")

    def save_model(self):
        if not os.path.exists("model_weights"):
            os.mkdir("model_weights")
        if not os.path.exists(self._checkpoint_path):
            os.mkdir(self._checkpoint_path)
        self.model.save_weights(self.checkpoint_filepath.format(epoch=0))

    def get_model(self):
        return tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(2,)),  # content_id and user_id
                tf.keras.layers.Dense(128, activation="relu"),  # 128 hidden layer
                tf.keras.layers.Dropout(0.2),  # dropout 20%
                tf.keras.layers.Dense(10),  # predict 10 things
            ]
        )


if __name__ == "__main__":
    mc = ModelController("untrained", True)
    mc.save_model()
