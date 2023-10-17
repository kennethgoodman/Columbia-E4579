import os
import torch
import time
import pandas as pd
import numpy as np
from src import db
from src.api.content.models import Content, GeneratedContentMetadata
from src.api.engagement.models import Engagement
from sklearn.preprocessing import OneHotEncoder, StandardScaler
#import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import pickle

# Get a list image sources based given list of content id
def fetch_database_data_by_contentid(content_id_list):
    try:
        return db.session.query(
            Content.id,
            Content.s3_bucket,
            Content.s3_id,
        ).filter(
            Content.id.in_(content_id_list)
        ).all()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Get a list image urls
def get_list_of_img_url(content_id_list):
    content_sources = fetch_database_data_by_contentid(content_id_list)
    image_urls_with_id = []
    if content_sources:
        for mid, s3_bucket, s3_id in content_sources:
            image_urls_with_id.append((mid, f"https://{s3_bucket}.s3.amazonaws.com/{s3_id}"))
    return image_urls_with_id

# show each images in list of image urls
def display_online_png(image_urls_with_id, user=False):
    if user:
        # Check if the folder exists
        if os.path.exists("engaged_images"):
            # Remove all the files in the folder
            for file in os.listdir("engaged_images"):
                os.remove("engaged_images/" + file)
    else:
        # Check if the folder exists
        if os.path.exists("recommendations_images"):
            # Remove all the files in the folder
            for file in os.listdir("recommendations_images"):
                os.remove("recommendations_images/" + file)
    for mid, url in image_urls_with_id:
        # Load the image from the URL
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        # Display the image using matplotlib
        #plt.imshow(img)
        #plt.axis('off')  # Hide axes
        #plt.title(f"Content ID: {mid}")  # Set window name as title
        ##plt.show(block=False) # show the image without block the thread
        ##plt.pause(1) # show the image for 1 second and then proceed
        #if user:
        #    os.makedirs("engaged_images", exist_ok=True)
        #    plt.savefig(f'engaged_images/content_id_{mid}.png')
        #else:
        #    os.makedirs("recommendations_images", exist_ok=True)
        #    plt.savefig(f'recommendations_images/content_id_{mid}.png')
        #plt.close()

# this the function to fetch the training data from the database
def fetch_data_by_user_id(user_id):
    try:
        return db.session.query(
            Engagement.user_id,
            Engagement.content_id,
            Engagement.engagement_type,
            Engagement.engagement_value,
            #Engagement.created_date,
            GeneratedContentMetadata.seed,
            GeneratedContentMetadata.guidance_scale,
            GeneratedContentMetadata.num_inference_steps,
            GeneratedContentMetadata.artist_style,
            GeneratedContentMetadata.source,
            GeneratedContentMetadata.model_version,
            GeneratedContentMetadata.prompt,
            GeneratedContentMetadata.prompt_embedding
        ).filter(
            Engagement.user_id == user_id
        ).join(
            GeneratedContentMetadata, Engagement.content_id == GeneratedContentMetadata.content_id
        ).all()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None



def generate_negative_samples(df, n_negatives_per_positive=1):
    """
    Generate negative samples for the dataset using an optimized approach.

    Parameters:
    - df: DataFrame containing the positive samples.
    - n_negatives_per_positive: Number of negative samples to generate per positive sample.

    Returns:
    - DataFrame with negative samples.
    """

    # Create an empty DataFrame to store the negative samples
    negative_samples = []

    user_groups = df.loc[(df.engagement_value == 1) | ((df.engagement_value > 700) & (df.engagement_value <= 2500))].groupby('user_id')['content_id'].unique()
    negative_engaged = df.loc[(df.engagement_value == -1) | (df.engagement_value <= 700) | (df.engagement_value > 2500)].groupby('user_id')['content_id'].unique()
    all_content = set(df['content_id'].unique())

    for user, engaged_content in user_groups.items():
        user_negative_engaged = negative_engaged.loc[user]
        # Find content that the user hasn't engaged with
        non_engaged_content = list(all_content - set(engaged_content) - set(user_negative_engaged))

        # Randomly select N content for negative samples
        num_negatives = min((len(engaged_content) - len(user_negative_engaged)) * n_negatives_per_positive, len(non_engaged_content))
        if num_negatives > 0:
            negative_content = np.random.choice(non_engaged_content, size=num_negatives, replace=False)
            # Create negative samples
            for content in negative_content:
                content_data = df.loc[df.content_id == content].iloc[0][['content_id', 'seed', 'guidance_scale',
                                                                         'num_inference_steps','artist_style', 'source',
                                                                         'model_version', 'prompt', 'prompt_embedding']]
                sample_data = {
                    'user_id': user,
                    'content_id': content,
                    'engagement_type': 'MillisecondsEngagedWith',
                    'engagement_value': 0,
                    'seed': content_data.seed,
                    'guidance_scale': content_data.guidance_scale,
                    'num_inference_steps': content_data.num_inference_steps,
                    'artist_style': content_data.artist_style,
                    'source': content_data.source,
                    'model_version': content_data.model_version,
                    'prompt': content_data.prompt,
                    'prompt_embedding': content_data.prompt_embedding,
                }
                negative_samples.append(sample_data)
        else:
            positive_content = np.random.choice(engaged_content, size=num_negatives*-1, replace=True)
            # Create negative samples
            for content in positive_content:
                content_data = df.loc[df.content_id == content].iloc[0][['content_id', 'seed', 'guidance_scale',
                                                                         'num_inference_steps','artist_style', 'source',
                                                                         'model_version', 'prompt', 'prompt_embedding']]
                sample_data = {
                    'user_id': user,
                    'content_id': content,
                    'engagement_type': 'MillisecondsEngagedWith',
                    'engagement_value': 800,
                    'seed': content_data.seed,
                    'guidance_scale': content_data.guidance_scale,
                    'num_inference_steps': content_data.num_inference_steps,
                    'artist_style': content_data.artist_style,
                    'source': content_data.source,
                    'model_version': content_data.model_version,
                    'prompt': content_data.prompt,
                    'prompt_embedding': content_data.prompt_embedding,
                }
                negative_samples.append(sample_data)

    # Convert list of dicts to DataFrame
    negative_df = pd.DataFrame(negative_samples)
    return negative_df

# create necessary tensors from databse
# you probably just need to call this function to get the tensor and train the model
# returns user_features_tensor, content_features_tensor, engagement_type_tensor, engagement_value_tensor
# from the professor's code
def create_training_tensors():
    data_list = fetch_database_train_data()
    if data_list:
        df = raw_database_data_to_df(data_list)
        negative_df_optimized = generate_negative_samples(df)
        df = pd.concat([df, negative_df_optimized], ignore_index=True)
        df.reset_index(drop=True)
        u_ts, c_ts, u_c, c_c, e_type_ts, e_val_ts = process_df_to_tensor(df)
        return u_ts, c_ts, u_c, c_c, e_type_ts, e_val_ts


# a helper function, turn list of tuples from database to a dataframe
def raw_database_data_to_df(raw_data):
    res = pd.DataFrame(raw_data,
                        columns=[
                            "user_id",
                            "content_id",
                            "engagement_type",
                            "engagement_value",
                            #"created_date",
                            "seed",
                            "guidance_scale",
                            "num_inference_steps",
                            "artist_style",
                            "source",
                            "model_version",
                            "prompt",
                            "prompt_embedding",
                        ])
    res['engagement_type'] = res['engagement_type'].apply(lambda val: val.name)
    return res

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

def get_tops(df, top_content=500):
    if os.path.isfile("/usr/src/app/src/recommendation_system/ml_models/foxtrot/assets/tops.pkl"):
        with open("/usr/src/app/src/recommendation_system/ml_models/foxtrot/assets/tops.pkl", "rb") as f:
            top_artist_styles, top_sources, top_seeds, top_n_content = pickle.load(f)
            return top_artist_styles, top_sources, top_seeds, top_n_content
    # Configuration options
    TOP_ARTIST_STYLES = 30
    TOP_SOURCES = 30
    TOP_SEEDS = 14
    TOP_CONTENT = top_content

    # Get the top artist styles, sources, and seeds
    top_artist_styles = df['artist_style'].value_counts().nlargest(TOP_ARTIST_STYLES).index.tolist()
    top_sources = df['source'].value_counts().nlargest(TOP_SOURCES).index.tolist()
    top_seeds = df['seed'].value_counts().nlargest(TOP_SEEDS).index.tolist()
    top_n_content = df.groupby('content_id')['engagement_value'].count().nlargest(TOP_CONTENT).index.tolist()
    with open("/usr/src/app/src/recommendation_system/ml_models/foxtrot/assets/tops.pkl", "wb") as file:
        pickle.dump((top_artist_styles, top_sources, top_seeds, top_n_content), file)
    return top_artist_styles, top_sources, top_seeds, top_n_content

def preprocess_for_tensor(df, top_artist_styles, top_sources, top_seeds, top_n_content, top_content=500):
    PROMPT_EMBEDDING_LENGTH = 512
    TOP_CONTENT = top_content

    # Replace less frequent artist styles, sources, and seeds with 'other'
    df['artist_style'] = df['artist_style'].apply(lambda x: x if x in top_artist_styles else 'other')
    df['source'] = df['source'].apply(lambda x: x if x in top_sources else 'other')
    df['seed'] = df['seed'].apply(lambda x: str(x) if x in top_seeds else 'other')

    # ensure the same columns(features) important!!!
    str_top_artist_styles = [str(i) for i in top_artist_styles]
    str_top_artist_styles.append('other')
    str_top_seeds = [str(i) for i in top_seeds]
    str_top_seeds.append('other')
    str_top_sources = [str(i) for i in top_sources]
    str_top_sources.append('other')
    str_model = [str(i) for i in list(df['model_version'].unique())]

    # One-hot encode categorical features
    categories = [str_top_artist_styles, str_model, str_top_seeds, str_top_sources]
    encoder = OneHotEncoder(categories=categories)
    content_onehot = encoder.fit_transform(df[['artist_style', 'model_version', 'seed', 'source']])
    content_onehot_df = pd.DataFrame(content_onehot.toarray(), columns=encoder.get_feature_names_out(['artist_style', 'model_version', 'seed', 'source']))
    df = pd.concat([df, content_onehot_df], axis=1)

    # Normalizing linear features
    scaler = StandardScaler()
    df[['guidance_scale', 'num_inference_steps']] = scaler.fit_transform(df[['guidance_scale', 'num_inference_steps']])

    # Compute top N content pieces based on engagement_value
    from collections import defaultdict
    user_vector_dict = defaultdict(lambda: {
        'millisecond_engaged_vector': np.zeros(len(top_n_content)),
        'like_vector': np.zeros(len(top_n_content)),
        'dislike_vector': np.zeros(len(top_n_content))
    })

    # Group by user_id and content_id, then apply the function
    engagement_aggregate = df[df['content_id'].isin(top_n_content)].groupby(['user_id', 'content_id']).apply(aggregate_engagement).reset_index()

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

    user_vector_df[millisecond_columns] = pd.DataFrame(user_vector_df['millisecond_engaged_vector'].tolist(), index=user_vector_df.index)
    user_vector_df[like_columns] = pd.DataFrame(user_vector_df['like_vector'].tolist(), index=user_vector_df.index)
    user_vector_df[dislike_columns] = pd.DataFrame(user_vector_df['dislike_vector'].tolist(), index=user_vector_df.index)

    # Drop the original vector columns
    user_vector_df.drop(['millisecond_engaged_vector', 'like_vector', 'dislike_vector'], axis=1, inplace=True)

    # Join User Vector To Df
    df = df.merge(
        user_vector_df.reset_index().rename(columns={'index': 'user_id'}),
        on='user_id',
        how='left',
    )
    df.fillna(0, inplace=True)
    del user_vector_df

    # Unpack prompt embedding
    prompt_columns = [f"prompt_embedding_{i}" for i in range(PROMPT_EMBEDDING_LENGTH)]
    df[prompt_columns] = pd.DataFrame(df['prompt_embedding'].tolist(), index=df.index)
    df = df.drop('prompt_embedding', axis=1)

    return df

def create_user_tensor(df, aggregate=False, top_content=500):
    if aggregate:
        df = df.groupby('user_id', as_index=False).sum()
    TOP_CONTENT = top_content
    # Create overall features for user
    user_columns = (
        [f'ms_engaged_{i}' for i in range(TOP_CONTENT)] +
        [f'like_vector_{i}' for i in range(TOP_CONTENT)] +
        [f'dislike_vector_{i}' for i in range(TOP_CONTENT)]
    )
    user_features = df[user_columns]
    user_features_tensor = torch.FloatTensor(user_features.values)
    del user_features

    return user_features_tensor, user_columns

def create_content_tensor(df, aggregate=False):
    if aggregate:
        df = df.groupby('content_id', as_index=False).sum()
    # Create overall features for content
    content_columns = (
        list(filter(lambda x: 'artist_style_' in x, df.columns)) +
        list(filter(lambda x: 'model_version_' in x, df.columns)) +
        list(filter(lambda x: 'source_' in x, df.columns)) +
        list(filter(lambda x: 'seed_' in x, df.columns)) +
        list(filter(lambda x: 'prompt_embedding_' in x, df.columns)) +
        ['content_id', 'guidance_scale', 'num_inference_steps']
    )
    content_features = df[content_columns]
    content_features_tensor = torch.FloatTensor(content_features.values)
    del content_features

    return content_features_tensor, content_columns


# same data process code from professor's code
def process_df_to_tensor(df):
    # Configuration options
    TOP_CONTENT = 500
    top_artist_styles, top_sources, top_seeds, top_n_content = get_tops(TOP_CONTENT)

    df = preprocess_for_tensor(df, top_artist_styles, top_sources, top_seeds, top_n_content, TOP_CONTENT)

    user_features_tensor, user_columns = create_user_tensor(df)

    content_features_tensor, content_columns = create_content_tensor(df)

    DISLIKE_ENGAGEMENT_TYPE_VALUE = 0
    LIKE_ENGAGEMENT_TYPE_VALUE = 1
    MS_ENGAGEMENT_TYPE_VALUE = 2

    engagement_type_value = df[['engagement_type', 'engagement_value']]
    engagement_type_value['engagement_type'] = engagement_type_value.apply(
        lambda x: (
           LIKE_ENGAGEMENT_TYPE_VALUE    if x['engagement_type'] == 'Like' and x['engagement_value'] ==  1 else # Like
           DISLIKE_ENGAGEMENT_TYPE_VALUE    if x['engagement_type'] == 'Like' and x['engagement_value'] == -1 else # Dislike
           MS_ENGAGEMENT_TYPE_VALUE                                                                          # MsEngagementType
        ), axis=1
    )
    engagement_type_tensor = torch.FloatTensor(engagement_type_value[['engagement_type']].values)
    engagement_value_tensor = torch.FloatTensor(engagement_type_value[['engagement_value']].values)
    del engagement_type_value
    del df
    return user_features_tensor, content_features_tensor, user_columns, content_columns, engagement_type_tensor, engagement_value_tensor

# content_ids of a user
# return both user likes and dislikes
def find_content_by_user(userID):
    raw_database_data = fetch_data_by_user_id(userID)
    df = raw_database_data_to_df(raw_database_data)
    liked_content = df[(df.engagement_value == 1)]['content_id'].tolist() # unique, len = 163
    disliked_content = df[(df.engagement_value == -1)]['content_id'].tolist() # unique, len = 196
    viewed_content = df[(df.engagement_value > 700) & (df.engagement_value <= 2500)]['content_id'].tolist() # not unique, len = 308
    engaged_content = set(liked_content + viewed_content)
    engaged_content = engaged_content - set(disliked_content)
    return engaged_content, set(disliked_content) # not unique; len = 408, unique vals = 404
