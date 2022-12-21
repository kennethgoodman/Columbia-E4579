import os
import lightgbm as lgbm
import pickle

class ModelController:
    def __init__(self, model_name, load_model=False):
        self.model_name = model_name
        self.model = self.get_model()
        if load_model:
            try:
                with open(self._model_path, 'rb') as f:
                    lgb_model = pickle.load(f)
                self.model = lgb_model
            except:
                pass  # no model to load from

    def train_model(self):
        pass

    @property
    def _model_path(self):
        return os.path.join(self.model_name + "_model.pkl")

    def save_model(self):
        pass

    def get_model(self):
        pass


if __name__ == "__main__":
    print(os.path)
    mc = ModelController("lgb", True)
    print(os.path.join(mc.model_name + "_model.pkl"))
    print(mc.model)
