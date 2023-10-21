
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
# import pickle
# import random

# Set up basic logging configuration
logging.basicConfig(level=logging.ERROR)

# Two Tower PyTorch Model
class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, content_dim, output_dim):
        super(TwoTowerModel, self).__init__()
        self.user_tower = nn.Linear(user_dim, output_dim)
        self.content_tower = nn.Linear(content_dim, output_dim)

    def forward_content(self, content_tensor):
        # Forward pass for content
        content_embedding = self.content_tower(content_tensor)
        return content_embedding

    def forward_user(self, user_tensor):
        # Forward pass for user
        user_embedding = self.user_tower(user_tensor)
        return user_embedding

    def forward(self, user_tensor, content_tensor):
        user_embedding = self.forward_user(user_tensor)
        content_embedding = self.forward_content(content_tensor)
        return user_embedding, content_embedding

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

class EngagementDataset(Dataset):
    def __init__(self, user_features, content_features, targets):
        self.user_features = user_features
        self.content_features = content_features
        self.targets = targets

    def __len__(self):
        return len(self.user_features)

    def __getitem__(self, index):
        return self.user_features[index], self.content_features[index], self.targets[index]

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, lambda_reg=0.01, lambda_orthog=0.01):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.lambda_reg = lambda_reg
        self.lambda_orthog = lambda_orthog

    def calculate_targets(self, engagement_type_vector, engagement_value_vector):
        # Conditions for 0=dislike, 1=like, and 2=milliseconds engaged
        DISLIKE_ENGAGEMENT_TYPE_VALUE = 0
        LIKE_ENGAGEMENT_TYPE_VALUE = 1
        MS_ENGAGEMENT_TYPE_VALUE = 2
        return torch.where(engagement_type_vector == DISLIKE_ENGAGEMENT_TYPE_VALUE,
                   torch.zeros_like(engagement_type_vector), # dislike
                   torch.where(engagement_type_vector == LIKE_ENGAGEMENT_TYPE_VALUE,
                        torch.ones_like(engagement_type_vector), # like
                        torch.where(engagement_value_vector < 500,
                            torch.zeros_like(engagement_type_vector), # bad engagement
                            torch.where(engagement_value_vector <= 2500,
                                        torch.ones_like(engagement_type_vector), # eng 500ms => 2.5s
                                        torch.zeros_like(engagement_type_vector), # bad engagement
                                        )
                                   ) # millisecond engaged with
                               )
                          )

    def forward(self, user_embedding, content_embedding, targets, with_debug=False):
        noise_factor = 0.0005
        user_embedding += noise_factor * torch.randn(*user_embedding.shape)
        content_embedding += noise_factor * torch.randn(*content_embedding.shape)

        cosine_sim = F.cosine_similarity(user_embedding, content_embedding, dim=1)

        # Contrastive loss
        loss_contrastive = torch.mean(
            (1 - targets) * torch.pow(cosine_sim, 2) +
            (targets)     * torch.pow(
                torch.clamp(self.margin - cosine_sim, min=0.0),
                2
            )
        )

        # Regularization terms
        reg_user = torch.norm(user_embedding, p=2)
        reg_content = torch.norm(content_embedding, p=2)
        regularization_loss = (reg_user + reg_content)

        # orthognal loss of content
        ortho_reg = torch.norm(
            torch.mm(content_embedding, content_embedding.t()) -
            torch.eye(content_embedding.size(0))
        )

        total_loss = (
            loss_contrastive +
            self.lambda_reg * regularization_loss +
            self.lambda_orthog * ortho_reg
        )
        if with_debug:
            print(f"""losses are:
              {loss_contrastive},
              {self.lambda_reg * regularization_loss},
              {self.lambda_orthog * ortho_reg},
              """)

        return total_loss

# Functions to convert DataFrame to Tensors


def preprocessing(df):
    # Configuration options
    TOP_ARTIST_STYLES = 30
    TOP_SOURCES = 30
    TOP_SEEDS = 14
    TOP_CONTENT = 251
    PROMPT_EMBEDDING_LENGTH = 512

    # Get the top artist styles, sources, and seeds
    top_artist_styles = df['artist_style'].value_counts().nlargest(TOP_ARTIST_STYLES).index.tolist()
    top_sources = df['source'].value_counts().nlargest(TOP_SOURCES).index.tolist()
    top_seeds = df['seed'].value_counts().nlargest(TOP_SEEDS).index.tolist()

    # Replace less frequent artist styles, sources, and seeds with 'other'
    df['artist_style'] = df['artist_style'].apply(lambda x: x if x in top_artist_styles else 'other')
    df['source'] = df['source'].apply(lambda x: x if x in top_sources else 'other')
    df['seed'] = df['seed'].apply(lambda x: str(x) if x in top_seeds else 'other')

    # One-hot encode categorical features
    encoder = OneHotEncoder()
    content_onehot = encoder.fit_transform(df[['artist_style', 'model_version', 'seed', 'source']])
    content_onehot_df = pd.DataFrame(content_onehot.toarray(), columns=encoder.get_feature_names_out(['artist_style', 'model_version', 'seed', 'source']))
    df = pd.concat([df, content_onehot_df], axis=1)

    # Normalizing linear features
    scaler = StandardScaler()
    df[['guidance_scale', 'num_inference_steps']] = scaler.fit_transform(df[['guidance_scale', 'num_inference_steps']])

    # Compute top N content pieces based on engagement_value
    from collections import defaultdict
    top_n_content = df.groupby('content_id')['engagement_value'].count().nlargest(TOP_CONTENT).index.tolist()
    user_vector_dict = defaultdict(lambda: {
        'millisecond_engaged_vector': np.zeros(len(top_n_content)),
        'like_vector': np.zeros(len(top_n_content)),
        'dislike_vector': np.zeros(len(top_n_content))
    })

    # Initialize vectors for each user
    def aggregate_engagement(group):
        # Summing millisecond engagement values
        millisecond_engagement_sum = group.loc[group['engagement_type'] != 'Like', 'engagement_value'].sum()

        # Counting likes and dislikes
        likes_count = group.loc[(group['engagement_type'] == 'Like') & (group['engagement_value'] == 1)].shape[0]
        dislikes_count = group.loc[(group['engagement_type'] == 'Like') & (group['engagement_value'] == -1)].shape[0]

        return pd.Series({
            'millisecond_engagement_sum': millisecond_engagement_sum,
            'likes_count': likes_count,
            'dislikes_count': dislikes_count
        })

    # Group by user_id and content_id, then apply the function
    engagement_aggregate = df[df['content_id'].isin(top_n_content)].groupby(['user_id', 'content_id']).apply(aggregate_engagement).reset_index()
    usr_avg = engagement_aggregate.groupby('user_id')['millisecond_engagement_sum'].sum().reset_index(name='user_avg_eng')
    cont_avg = engagement_aggregate.groupby('content_id')['millisecond_engagement_sum'].sum().reset_index(name='cont_avg_eng')

    df = pd.merge(df, usr_avg, on=['user_id'], how='left').fillna(0)
    df = pd.merge(df, cont_avg, on=['content_id'], how='left').fillna(0)

    # Now, populate your user_vector_dict
    for _, row in engagement_aggregate.iterrows():
        user_id = row['user_id']
        content_id = row['content_id']
        idx = top_n_content.index(content_id)

        user_vector_dict[user_id]['millisecond_engaged_vector'][idx] = row['millisecond_engagement_sum']
        user_vector_dict[user_id]['like_vector'][idx] = row['likes_count']
        user_vector_dict[user_id]['dislike_vector'][idx] = row['dislikes_count']

    # Convert to DataFrame
    user_vector_df = pd.DataFrame.from_dict(user_vector_dict, orient='index')
    del user_vector_dict

    # Unpack vector columns into individual columns
    millisecond_columns = [f"ms_engaged_{i}" for i in range(TOP_CONTENT)]
    like_columns = [f"like_vector_{i}" for i in range(TOP_CONTENT)]
    dislike_columns = [f"dislike_vector_{i}" for i in range(TOP_CONTENT)]

    # [:TOP_CONTENT] is WRONG but doing it to get code to work
    user_vector_df[millisecond_columns] = pd.DataFrame(user_vector_df['millisecond_engaged_vector'].tolist().apply(lambda x: x[:TOP_CONTENT]), index=user_vector_df.index)
    user_vector_df[like_columns] = pd.DataFrame(user_vector_df['like_vector'].tolist().apply(lambda x: x[:TOP_CONTENT]), index=user_vector_df.index)
    user_vector_df[dislike_columns] = pd.DataFrame(user_vector_df['dislike_vector'].tolist().apply(lambda x: x[:TOP_CONTENT]), index=user_vector_df.index)

    # Drop the original vector columns
    user_vector_df.drop(['millisecond_engaged_vector', 'like_vector', 'dislike_vector'], axis=1, inplace=True)

    # Join User Vector To Df
    df = df.merge(
        user_vector_df.reset_index().rename(columns={'index': 'user_id'}),
        on='user_id'
    )
    del user_vector_df

    # Unpack prompt embedding
    prompt_columns = [f"prompt_embedding_{i}" for i in range(PROMPT_EMBEDDING_LENGTH)]
    df[prompt_columns] = pd.DataFrame(df['prompt_embedding'].tolist(), index=df.index)
    df = df.drop('prompt_embedding', axis=1)

    return df


def df_to_content_tensor(df):
    # Group by content_id and sum
    df = preprocessing(df)
    aggregated = df.groupby('content_id').sum()
    content_tensor = torch.tensor(aggregated.values, dtype=torch.float32)
    return content_tensor

def df_to_user_tensor(df):
    # Group by user_id and sum
    df = preprocessing(df)
    aggregated = df.groupby('user_id').sum()
    user_tensor = torch.tensor(aggregated.values, dtype=torch.float32)
    return user_tensor


# Model Wrapper
class ModelWrapper:
    def __init__(self, model_path="/usr/src/app/src/recommendation_system/ml_models/charlie/two_tower_trained_saved.pth"):
        if not model_path:
            self.model = DummyTwoTowerModel()
        else:
            self.model = TwoTowerModel(754, 594, 64)
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
