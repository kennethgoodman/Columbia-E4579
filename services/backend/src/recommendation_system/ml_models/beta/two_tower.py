
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
legalize = lambda s:os.path.join(script_dir, s)

# Set up basic logging configuration
logging.basicConfig(level=logging.ERROR)

# Two Tower PyTorch Model
class TwoTowerModel(nn.Module):
    def __init__(self):
        super(TwoTowerModel, self).__init__()
        self.user_tower = nn.Linear(753, 64)
        self.content_tower = nn.Linear(592, 64)

    def forward_content(self, content_tensor):
        content_embedding = self.content_tower(content_tensor)
        return content_embedding
        

    def forward_user(self, user_tensor):
        user_embedding = self.user_tower(user_tensor)
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
    from collections import defaultdict
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import pickle
    import json

    TOP_ARTIST_STYLES = 30
    TOP_SOURCES = 30
    TOP_SEEDS = 14
    PROMPT_EMBEDDING_LENGTH = 512


    df['seed'] = df.seed.apply(str)

    with open(legalize("clip_embed.pkl"), "rb") as f:
        clip_embed = pickle.load(f)
        
    clip_embedding = pd.read_csv(legalize('clip_lookup.csv'))
    clip_embedding['vecs'] = list(clip_embed)

    with open(legalize('top_data.json'), 'r') as file:
        top_data = json.load(file)
    top_data['top_artist_styles'].append('other')
    top_data['top_sources'].append('other')
    top_data['top_seeds'].append('other')

    #print('sucess loading content data')

    df = df.groupby('content_id').agg({'artist_style':lambda x: x.iloc[0],
                                      'source':lambda x: x.iloc[0],
                                      'seed':lambda x: x.iloc[0],
                                      'guidance_scale':lambda x: x.iloc[0],
                                      'num_inference_steps':lambda x: x.iloc[0],
                                      'content_id':lambda x:x.iloc[0]})
    
    clip_e = []
    
    for i in df.content_id:
        if i in clip_embedding.content_id:
            clip_embedding[clip_embedding.content_id==i].vecs.iloc[0].astype(np.float32)
        else:
            clip_e.append(np.full((PROMPT_EMBEDDING_LENGTH,), 0, dtype=np.float32))
            
    clip_e = np.array(clip_e)

    #print('Success constructed clip_e')
    
    NUM = {
        'artist_style':30,
        'seed':14,
        'source':30
    }

    top_artist_styles = top_data['top_artist_styles']
    top_sources = top_data['top_sources']
    top_seeds = [str(_) for _ in top_data['top_seeds']]


    # Replace less frequent artist styles, sources, and seeds with 'other'
    df['artist_style'] = df['artist_style'].apply(lambda x: x if x in top_artist_styles else 'other')
    df['source'] = df['source'].apply(lambda x: x if x in top_sources else 'other')
    df['seed'] = df['seed'].apply(lambda x: str(x) if x in top_seeds else 'other')

    
    
    stringify = lambda lst : [str(_) for _ in lst]
    content_onehot = [np.array(
    [[(top == df[column].iloc[i]) for top in stringify(top_data[f'top_{column}s'])] for i in range(len(df))]).astype(np.float32)
           for column in ['artist_style','source','seed']]
    content_onehot = np.hstack(content_onehot)

    print('sucess onehot for artist_style etc')

    # Normalizing linear features
    scaler = StandardScaler()
    df[['guidance_scale', 'num_inference_steps']] = scaler.fit_transform(df[['guidance_scale', 'num_inference_steps']])

    content_columns2 = ['content_id','guidance_scale', 'num_inference_steps']
    content_features2 = df[content_columns2]

    print('success subsetting df')
    
    content_onehott = content_onehot.astype(np.float32)
    clip_e = clip_e.astype(np.float32)
    content_f2 = content_features2.values.astype(np.float32)

    

    content_features_tensor = torch.FloatTensor(
        np.hstack([content_onehott, clip_e, content_f2])
    ) 
    print(f'Sucess creating array {content_features_tensor.shape}')
    return content_features_tensor

def df_to_user_tensor(df):
    from collections import defaultdict
    import json

    
    TOP_CONTENT = 251
    with open(legalize('top_n_content.json'), 'r') as file:
        top_n_content = json.load(file)
    
    user_columns = (
        [f'ms_engaged_{i}' for i in range(TOP_CONTENT)] +
        [f'like_vector_{i}' for i in range(TOP_CONTENT)] +
        [f'dislike_vector_{i}' for i in range(TOP_CONTENT)]
    )
    print('Successfully loaded data')
    
    # User: Construct groupby-like columns for each user using
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
    try:
        engagement_aggregate = df[df['content_id'].isin(top_n_content)].groupby(['user_id', 'content_id'],group_keys=False).apply(aggregate_engagement).reset_index()

        #print(f'Successfully aggregated, {engagement_aggregate.shape}')
        user_vector_dict = defaultdict(lambda: {
            'millisecond_engaged_vector': np.zeros(len(top_n_content)),
            'like_vector': np.zeros(len(top_n_content)),
            'dislike_vector': np.zeros(len(top_n_content))
        })
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
        
        user_vector_mil = np.vstack(user_vector_df['millisecond_engaged_vector']).astype(np.float32)
        user_vector_like = np.vstack(user_vector_df['like_vector']).astype(np.float32)
        user_vector_dislike = np.vstack(user_vector_df['dislike_vector']).astype(np.float32)
        
        #print('Sucessfully constructed user_vectors')
        user_features_tensor = torch.FloatTensor(
            np.hstack([user_vector_mil,user_vector_like,user_vector_dislike])
        )
    except:
        user_features_tensor = torch.ones((753,), dtype=torch.float32)

    return user_features_tensor
# Model Wrapper
class ModelWrapper:
    def __init__(self, model_path="clip_model.pth"):
        if not model_path:
            self.model = DummyTwoTowerModel()
        else:
            self.model = TwoTowerModel()
            
            self.model.load_state_dict(torch.load(legalize(model_path), map_location=torch.device('cpu')))
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
