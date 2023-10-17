import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
from src.recommendation_system.ml_models.foxtrot.utils import (
    get_tops,
    preprocess_for_tensor,
    create_user_tensor,
    create_content_tensor,
)

# Set up basic logging configuration
logging.basicConfig(level=logging.ERROR)

# Two Tower PyTorch Model
class TwoTowerModel(nn.Module):
    def __init__(self, user_dim=1500, content_dim=593, hidden_dim=512, output_dim=64):
        super(TwoTowerModel, self).__init__()

        self.dropout = nn.Dropout(0.5)
        # User tower
        self.user_linear1 = nn.Linear(user_dim, hidden_dim)
        self.user_relu = nn.LeakyReLU()
        self.user_batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.user_batchnorm2 = nn.BatchNorm1d(hidden_dim)
        self.user_linear2 = nn.Linear(hidden_dim, output_dim)

        # Content tower
        self.content_linear1 = nn.Linear(content_dim, hidden_dim)
        self.content_relu = nn.LeakyReLU()
        self.content_batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.content_batchnorm2 = nn.BatchNorm1d(hidden_dim)
        self.content_linear2 = nn.Linear(hidden_dim, output_dim)

        # initialization
        nn.init.kaiming_normal_(self.user_linear1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.content_linear1.weight, mode='fan_in', nonlinearity='relu')

    def forward_content(self, content_tensor):
        # Content tower
        content_embedding = self.content_linear1(content_tensor)
        content_embedding = self.content_relu(content_embedding)
        content_embedding = self.content_batchnorm1(content_embedding)
        content_embedding = self.dropout(content_embedding)
        content_embedding = self.content_linear2(content_embedding)

        return content_embedding

    def forward_user(self, user_tensor):
        # User tower
        user_embedding = self.user_linear1(user_tensor)
        user_embedding = self.user_relu(user_embedding)
        user_embedding = self.user_batchnorm1(user_embedding)
        user_embedding = self.dropout(user_embedding)
        user_embedding = self.user_linear2(user_embedding)

        return user_embedding


# Dummy Two Tower PyTorch Model for testing
class DummyTwoTowerModel(nn.Module):
    def __init__(self):
        super(DummyTwoTowerModel, self).__init__()

    def forward_content(self, content_tensor):
        # Return dummy embeddings of shape (content_tensor length, 64)
        return torch.randn((len(content_tensor), 64))

    def forward_user(self, user_tensor):
        # Return dummy embeddings of shape (user_tensor length, 64)
        return torch.randn((len(user_tensor), 64))


# Functions to convert DataFrame to Tensors
def df_to_content_tensor(df):
    top_artist_styles, top_sources, top_seeds, top_n_content = get_tops(df)
    df = preprocess_for_tensor(df, top_artist_styles, top_sources, top_seeds, top_n_content)
    content_tensor, _ = create_content_tensor(df, True)
    return content_tensor

def df_to_user_tensor(df):
    top_artist_styles, top_sources, top_seeds, top_n_content = get_tops()
    df = preprocess_for_tensor(df, top_artist_styles, top_sources, top_seeds, top_n_content)
    user_tensor, _ = create_user_tensor(df, True)
    return user_tensor

# Model Wrapper
class ModelWrapper:
    def __init__(self, model_path="/usr/src/app/src/recommendation_system/ml_models/foxtrot/model.pth"):
        if not model_path:
            self.model = DummyTwoTowerModel()
        else:
            self.model = TwoTowerModel()
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

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

# Initialize ModelWrapper with an empty path to return a dummy model for testing
model_wrapper = ModelWrapper()
