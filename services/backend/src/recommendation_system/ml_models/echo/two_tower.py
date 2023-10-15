
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")
import pickle
import joblib
file_path = '/usr/src/app/src/recommendation_system/ml_models/echo/'
top_artist_styles = joblib.load(file_path+'top_artist_styles.pkl')
top_sources = joblib.load(file_path+'top_sources.pkl')
top_seeds = joblib.load(file_path+'top_seeds.pkl')
encoder = joblib.load(file_path+'encoder.pkl')
scaler = joblib.load(file_path+'scaler.pkl')


# Set up basic logging configuration
logging.basicConfig(level=logging.ERROR)

TOP_ARTIST_STYLES = 30
TOP_SOURCES = 30
TOP_SEEDS = 14
TOP_CONTENT = 251
PROMPT_EMBEDDING_LENGTH = 512
with open(file_path + 'top_n_content.pkl','rb') as file:
  top_n_content = pickle.load(file)


# Two Tower PyTorch Model
class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, content_dim, output_dim):
        super(TwoTowerModel, self).__init__()
        # Define the layers for content and user towers here if needed
        # For the sake of simplicity, I'm leaving it empty now
        self.user_tower = nn.Linear(user_dim, output_dim)
        self.content_tower = nn.Linear(content_dim, output_dim)

    def forward_content(self, content_tensor):
        # Forward pass for content
        return self.content_tower(content_tensor)

    def forward_user(self, user_tensor):
        # Forward pass for user
        return self.user_tower(user_tensor)

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

def user_preprocessing(df):
    df['engagement_type'] = df['engagement_type'].apply(lambda x: x.name)    
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
    for user_id in df['user_id'].unique():
        _ = user_vector_dict[user_id]
    if len(df[df['content_id'].isin(top_n_content)]) != 0:
        engagement_aggregate = df[df['content_id'].isin(top_n_content)].groupby(['user_id', 'content_id']).apply(aggregate_engagement).reset_index()
        for _, row in engagement_aggregate.iterrows():
            user_id = row['user_id']
            content_id = row['content_id']
            idx = top_n_content.index(content_id)

            user_vector_dict[user_id]['millisecond_engaged_vector'][idx] = row['millisecond_engagement_sum']
            user_vector_dict[user_id]['like_vector'][idx] = row['likes_count']
            user_vector_dict[user_id]['dislike_vector'][idx] = row['dislikes_count']

    # Convert to DataFrame
    user_vector_df = pd.DataFrame.from_dict(user_vector_dict, orient='index')
    millisecond_columns = [f"ms_engaged_{i}" for i in range(TOP_CONTENT)]
    like_columns = [f"like_vector_{i}" for i in range(TOP_CONTENT)]
    dislike_columns = [f"dislike_vector_{i}" for i in range(TOP_CONTENT)]

    user_vector_df[millisecond_columns] = pd.DataFrame(user_vector_df['millisecond_engaged_vector'].tolist(), index=user_vector_df.index)
    user_vector_df[like_columns] = pd.DataFrame(user_vector_df['like_vector'].tolist(), index=user_vector_df.index)
    user_vector_df[dislike_columns] = pd.DataFrame(user_vector_df['dislike_vector'].tolist(), index=user_vector_df.index)
    # Drop the original vector columns
    user_vector_df.drop(['millisecond_engaged_vector', 'like_vector', 'dislike_vector'], axis=1, inplace=True)

    # Join User Vector To Df
    df = df.merge(
        user_vector_df.reset_index().rename(columns={'index': 'user_id'}),
        on='user_id'
    )

    user_columns = (
        ['user_id'] + 
        [f'ms_engaged_{i}' for i in range(TOP_CONTENT)] +
        [f'like_vector_{i}' for i in range(TOP_CONTENT)] +
        [f'dislike_vector_{i}' for i in range(TOP_CONTENT)]
    )
    user_features = df[user_columns]
    return user_features

def content_preprocessing(df):

    # Get the top artist styles, sources, and seeds
    df['model_version'] = df['model_version'].apply(lambda x:eval(x))
    df['artist_style'] = df['artist_style'].apply(lambda x: x if x in top_artist_styles else 'other')
    df['source'] = df['source'].apply(lambda x: x if x in top_sources else 'other')
    df['seed'] = df['seed'].apply(lambda x: str(x) if x in top_seeds else 'other')
    feature_names_columns = []
    content_onehot = encoder.transform(df[['artist_style', 'model_version', 'seed', 'source']])
    for i,name in enumerate(['artist_style', 'model_version', 'seed', 'source']):
        for col in encoder.categories_[i]:
            if not isinstance(col,str):
                col = str(col)
            feature_names_columns.append(name+'_'+col)
    content_onehot_df = pd.DataFrame(content_onehot.toarray(), columns=feature_names_columns) # encoder.get_feature_names_out(['artist_style', 'model_version', 'seed', 'source'])
    df = pd.concat([df, content_onehot_df], axis=1)

    # Normalizing linear features
    df[['guidance_scale', 'num_inference_steps']] = scaler.transform(df[['guidance_scale', 'num_inference_steps']])

    # Unpack prompt embedding
    prompt_columns = [f"prompt_embedding_{i}" for i in range(PROMPT_EMBEDDING_LENGTH)]
    df[prompt_columns] = pd.DataFrame(df['prompt_embedding'].tolist(), index=df.index)
    df = df.drop('prompt_embedding', axis=1)

    content_columns = (
    list(filter(lambda x: 'artist_style_' in x, df.columns)) +
    list(filter(lambda x: 'model_version_' in x, df.columns)) +
    list(filter(lambda x: 'source_' in x, df.columns)) +
    list(filter(lambda x: 'seed_' in x, df.columns)) +
    list(filter(lambda x: 'prompt_embedding_' in x, df.columns)) +
    ['content_id', 'guidance_scale', 'num_inference_steps']
)
    content_features = df[content_columns]

    return content_features

# Functions to convert DataFrame to Tensors
def df_to_content_tensor(df):
    # Group by content_id and sum

    aggregated = df.groupby('content_id').sum().reset_index()
    content_tensor = torch.tensor(aggregated.values, dtype=torch.float32)
    return content_tensor

def df_to_user_tensor(df):
    # Group by user_id and sum
    # df.drop_duplicates(inplace = True)
    aggregated = df.groupby('user_id').mean()
    user_tensor = torch.tensor(aggregated.values, dtype=torch.float32)
    return user_tensor

# Model Wrapper
class ModelWrapper:
    def __init__(self, model_path='/usr/src/app/src/recommendation_system/ml_models/echo/model_1.pt'):
        if not model_path:
            self.model = DummyTwoTowerModel()
        else:
            self.model = TwoTowerModel(753,593,64)
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def generate_content_embeddings(self, df):
        content_features= content_preprocessing(df)
        content_tensor = df_to_content_tensor(content_features)

        if len(df["content_id"].unique()) != len(content_tensor):
            logging.error("Mismatch in content tensor length")
            return np.array([])

        embeddings = self.model.forward_content(content_tensor).detach().numpy().astype(np.float32)
        if len(embeddings) != len(content_tensor) or embeddings.shape[1] > 64:
            logging.error("Mismatch in embeddings and tensor length or embedding size exceeds 64")
            return np.array([])

        return embeddings

    def generate_user_embeddings(self, df):
        user_features = user_preprocessing(df)
        user_tensor = df_to_user_tensor(user_features)
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