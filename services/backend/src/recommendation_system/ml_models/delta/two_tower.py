import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.ERROR)

class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, content_dim, output_dim):
        super(TwoTowerModel, self).__init__()
        # Define the layers for content and user towers here if needed
        self.fc0 = nn.Linear(user_dim, output_dim)
        self.fc0_1= nn.Linear(content_dim, output_dim)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

    def forward(self, user_tensor, content_tensor):
        return self.forward_user(user_tensor), self.forward_content(content_tensor)

    def forward_content(self, content_tensor):
        content_tensor = F.relu(self.fc0_1(content_tensor))
        content_tensor = F.relu(self.fc1(content_tensor))
        content_tensor = F.relu(self.fc2(content_tensor))
        content_tensor = self.fc3(content_tensor)

        return content_tensor


    def forward_user(self, user_tensor):
        user_tensor = F.relu(self.fc0(user_tensor))
        user_tensor = F.relu(self.fc1(user_tensor))
        user_tensor = F.relu(self.fc2(user_tensor))
        user_tensor = self.fc3(user_tensor)

        return user_tensor


class DummyTwoTowerModel(nn.Module):
    def __init__(self):
        super(DummyTwoTowerModel, self).__init__()

    def forward_content(self, content_tensor):
        # Return dummy embeddings of shape (content_tensor length, 64)
        return torch.randn((len(content_tensor), 64))

    def forward_user(self, user_tensor):
        # Return dummy embeddings of shape (user_tensor length, 64)
        return torch.randn((len(user_tensor), 64))

def df_to_content_tensor(df):
    aggregated = df.groupby('content_id').sum()
    content_ids = aggregated.index
    content_values = aggregated.values

    num_repeats = max(1, 50000 // len(content_ids))

    repeated_content_ids = content_ids.repeat(num_repeats)
    repeated_content_values = content_values.repeat(num_repeats, axis=0)

    repeated_content_ids = repeated_content_ids[:50000]
    repeated_content_values = repeated_content_values[:50000]

    content_tensor = torch.tensor(repeated_content_values, dtype=torch.float32)

    return content_tensor

def df_to_user_tensor(df):
    aggregated = df.groupby('user_id').sum()
    user_tensor = torch.tensor(aggregated.values, dtype=torch.float32)
    return user_tensor


class ModelWrapper:
    def __init__(self, model_path="/usr/src/app/src/recommendation_system/ml_models/delta/twotower_model.dict"):
        if not model_path:
            self.model = DummyTwoTowerModel()
        else:
            self.model = TwoTowerModel(753, 593, 64)
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def generate_content_embeddings(self, df):
        content_tensor = df_to_content_tensor(df)
        if len(df["content_id"].unique()) != len(content_tensor):
            logging.error("Mismatch in content tensor length")
            return np.array([])

        embeddings = self.model.forward_content(content_tensor).detach().numpy().astype(np.float32)
        if len(embeddings) != len(content_tensor) or embeddings.shape[1] > 64:
            logging.error("Mismatch in embeddings and tensor length or embedding size exceeds 64")
            return np.array([])

        return embeddings

    def generate_user_embeddings(self, df):
        user_tensor = df_to_user_tensor(df)
        if len(df["user_id"].unique()) != len(user_tensor):
            logging.error("Mismatch in user tensor length")
            return np.array([])

        embeddings = self.model.forward_user(user_tensor).detach().numpy().astype(np.float32)
        if len(embeddings) != len(user_tensor) or embeddings.shape[1] > 64:
            logging.error("Mismatch in embeddings and tensor length or embedding size exceeds 64")
            return np.array([])

        return embeddings

model_wrapper = ModelWrapper()
