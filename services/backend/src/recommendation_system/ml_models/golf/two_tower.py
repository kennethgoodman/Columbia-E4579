
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging

# Set up basic logging configuration
logging.basicConfig(level=logging.ERROR)

# Two Tower PyTorch Model
class TwoTowerModel(nn.Module):
    def __init__(self):
        super(TwoTowerModel, self).__init__()
        # Define the layers for content and user towers here if needed
        # For the sake of simplicity, I'm leaving it empty now

    def forward_content(self, content_tensor):
        # Forward pass for content
        raise NotImplementedError("forward_content needs to be implemented")

    def forward_user(self, user_tensor):
        # Forward pass for user
        raise NotImplementedError("forward_user needs to be implemented")

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
    # Group by content_id and sum
    aggregated = df.groupby('content_id').sum()
    content_tensor = torch.tensor(aggregated.values, dtype=torch.float32)
    return content_tensor

def df_to_user_tensor(df):
    # Group by user_id and sum
    aggregated = df.groupby('user_id').sum()
    user_tensor = torch.tensor(aggregated.values, dtype=torch.float32)
    return user_tensor

# Model Wrapper
class ModelWrapper:
    def __init__(self, model_path=""):
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
