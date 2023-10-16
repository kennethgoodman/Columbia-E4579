import torch
import mrpt
import pandas as pd
from src import db
from src.recommendation_system.ml_models.foxtrot.two_tower import ModelWrapper
from src.api.engagement.models import Engagement
from src.api.content.models import Content, GeneratedContentMetadata
from sqlalchemy.sql.expression import func
#import matplotlib.pyplot as plt

def fetch_data_stub():
    try:
        return db.session.query(
            Engagement.content_id,
            Engagement.user_id,
            Engagement.engagement_type,
            Engagement.engagement_value,
            GeneratedContentMetadata.seed,
            GeneratedContentMetadata.guidance_scale,
            GeneratedContentMetadata.num_inference_steps,
            GeneratedContentMetadata.artist_style,
            GeneratedContentMetadata.source,
            GeneratedContentMetadata.model_version,
            GeneratedContentMetadata.prompt_embedding
        ).join(
            GeneratedContentMetadata, Engagement.content_id == GeneratedContentMetadata.content_id
        )
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def instantiate_indexes(base_data_limit):
    res = None
    index_to_content_id = {}
    content_id_to_index = {}
    try:
        distinct_content_ids_subquery = db.session.query(
            Content.id
        ).order_by(func.random()).limit(base_data_limit).subquery()
        contents = fetch_data_stub().join(
            distinct_content_ids_subquery, distinct_content_ids_subquery.c.id == Engagement.content_id
        ).order_by(Engagement.content_id).all()
        df = pd.DataFrame(contents)

        index_to_content_id = df['content_id'].to_dict()
        content_id_to_index = {v: k for k, v in index_to_content_id.items()}
        model = ModelWrapper()

        try:
            data = model.generate_content_embeddings(df)
            if len(data) < 101:
                index = None
            else:
                index = mrpt.MRPTIndex(data)
                index.build_autotune_sample(0.9, 10)
                res = index
        except Exception as e:
            print(f"Error during index instantiation for foxtrot, {e}")
            index = None

    except Exception as e:
        print(f"Error during index instantiation: {e}")
    return res, index_to_content_id, content_id_to_index

def get_ANN_recommednations(embedding, indexes, index_to_content_id, K):
    try:
        K = min(100, K)
        similar_indices, scores = indexes.ann(embedding, k=K, return_distances=True)
        new_similar_content, new_scores = [], []
        for idx, score in zip(similar_indices[0], scores[0]):
            if idx != -1:
                new_similar_content.append(index_to_content_id[idx])
                new_scores.append(score)
    except Exception as e:
        print(f"Error during ANN recommendations: {e}")
        return [], []
    return new_similar_content, new_scores




def get_recommendations_from_user(user_id, base_data_length=1000):
    try:
        # Fetch engagements for user
        indexes, index_to_content_id, _ = instantiate_indexes(base_data_length)
        user_engagements = fetch_data_stub().filter(
            Engagement.user_id == user_id
        ).all()

        user_df = pd.DataFrame(user_engagements)
        model = ModelWrapper()

        user_embedding = model.generate_user_embeddings(user_df)

        if len(user_embedding) == 0:
            return [], []

        return get_ANN_recommednations(user_embedding, indexes, index_to_content_id, 100)
    except Exception as e:
        print(f"Error during ANN recommendations: {e}")
        return [], []

def get_recommendations_from_content(content_id, base_data_length=1000):
    try:
        # Fetch engagements for user
        indexes, index_to_content_id, _ = instantiate_indexes(base_data_length)
        user_engagements_for_content = fetch_data_stub().filter(
            Content.id == content_id
        ).all()

        content_df = pd.DataFrame(user_engagements_for_content)
        model = ModelWrapper()

        content_embedding = model.generate_content_embeddings(content_df)

        if len(content_embedding) == 0:
            return [], []

        return get_ANN_recommednations(content_embedding, indexes, index_to_content_id, 100)
    except Exception as e:
        print(f"Error during ANN recommendations: {e}")
        return [], []

# Some Plotting
def random_cosine_similarity(content_embedding, n=20):
    similarity = torch.nn.functional.cosine_similarity(content_embedding.unsqueeze(0), content_embedding.unsqueeze(1), dim=2)
    # Ensure that the diagonal contains -inf so it won't interfere with the upper triangular part
    similarity = similarity - 2*torch.eye(similarity.shape[0])  # Subtracting 2 will give -2 for the diagonal, which is smaller than any possible cosine similarity (-1 to 1).

    # Get the upper triangular part without the diagonal
    indices = torch.triu_indices(row=similarity.shape[0], col=similarity.shape[1], offset=1)
    upper_triangular_part = similarity[indices[0], indices[1]]
    return upper_triangular_part[torch.randperm(upper_triangular_part.size(0))[:n]]


def plot_bins(content_embedding):
    similarity = torch.nn.functional.cosine_similarity(content_embedding.unsqueeze(0), content_embedding.unsqueeze(1), dim=2)
    # Ensure that the diagonal contains -inf so it won't interfere with the upper triangular part
    similarity = similarity - 2*torch.eye(similarity.shape[0])  # Subtracting 2 will give -2 for the diagonal, which is smaller than any possible cosine similarity (-1 to 1).

    # Get the upper triangular part without the diagonal
    indices = torch.triu_indices(row=similarity.shape[0], col=similarity.shape[1], offset=1)
    upper_triangular_part = similarity[indices[0], indices[1]]
    bins = 100
    x = range(-100, 100, 2)
    actual_hist = torch.histc(upper_triangular_part, bins=100, min=-1, max=1)
    actual_hist /= actual_hist.sum() # normalized
    #plt.bar(x, actual_hist, align='center')
    #plt.xlabel('Bins')
    #plt.show(block=False)
    #plt.pause(1)
    #plt.close()


