from src import db
from sqlalchemy.sql.expression import func
from src.api.content.models import Content, GeneratedContentMetadata
from src.api.engagement.models import Engagement
from flask import current_app
import mrpt
import pandas as pd
import traceback

# Global Variables
INDEXES = {}
index_to_content_id = {}
content_id_to_index = {}
team_wrappers = {}

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
        print(traceback.format_exc())
        return None

def instantiate_indexes():
    try:
        if current_app.config.get("NUMBER_OF_CONTENT_IN_ANN") == 0:
            return
        distinct_content_ids_subquery = db.session.query(
            Content.id
        ).order_by(func.random()).limit(current_app.config.get("NUMBER_OF_CONTENT_IN_ANN")).subquery()
        contents = fetch_data_stub().join(
            distinct_content_ids_subquery, distinct_content_ids_subquery.c.id == Engagement.content_id
        ).order_by(Engagement.content_id).all()
        df = pd.DataFrame(contents)

        global team_wrappers
        teams = current_app.config.get("TEAMS_TO_RUN_FOR")
        for team in teams:
            module_path = f"src.recommendation_system.ml_models.{team}.two_tower"
            try:
                ModelWrapper = __import__(module_path, fromlist=['ModelWrapper']).ModelWrapper
                team_wrappers[team] = ModelWrapper()
                print(f"Done ModelWrapper instantiation for {team} successfully")
            except Exception as e:
                print(f"Error during ModelWrapper instantiation for {team}, {e}")
                print(traceback.format_exc())
                team_wrappers[team] = None

        global index_to_content_id, content_id_to_index, INDEXES
        index_to_content_id = df['content_id'].to_dict()
        content_id_to_index = {v: k for k, v in index_to_content_id.items()}

        for team in teams:
            try:
                data = team_wrappers[team].generate_content_embeddings(df)
                if len(data) < 101:
                    raise ValueError(f"len(data) == {len(data)} < 101")
                    index = None
                else:
                    index = mrpt.MRPTIndex(data)
                    index.build_autotune_sample(0.9, 200)
                print(f"Done index instantiation for {team} successfully")
            except Exception as e:
                print(f"Error during index instantiation for {team}, {e}")
                print(traceback.format_exc())
                index = None
            INDEXES[team] = index
    except Exception as e:
        print(f"Error during index instantiation: {e}")
        print(traceback.format_exc())

def get_ANN_recommednations(embedding, team, K):
    try:
        K = min(100, K)
        global index_to_content_id, INDEXES
        if INDEXES[team] is None:
            return [], []
        similar_indices, scores = INDEXES[team].ann(embedding, k=K, return_distances=True)
        new_similar_content, new_scores = [], []
        for idx, score in zip(similar_indices[0], scores[0]):
            if idx != -1:
                new_similar_content.append(index_to_content_id[idx])
                new_scores.append(score) 
        return new_similar_content, new_scores
    except Exception as e:
        print(f"Error during get_ANN_recommednations recommendations for {team}: {e}")
        print(traceback.format_exc())
        return [], []

def get_ANN_recommendations_from_user(user_id, team, K):
    try:
        # Fetch engagements for user
        user_engagements = fetch_data_stub().filter(
            Engagement.user_id == user_id
        ).all()

        user_df = pd.DataFrame(user_engagements)
        
        if len(user_df) == 0:
            return [], []

        user_embedding = team_wrappers[team].generate_user_embeddings(user_df)

        if len(user_embedding) == 0:
            return [], []

        return get_ANN_recommednations(user_embedding, team, K)
    except Exception as e:
        print(f"Error during get_ANN_recommendations_from_user recommendations for {team}: {e}")
        print(traceback.format_exc())
        return [], []

def get_ANN_recommendations_from_content(content_id, team, K):
    try:
        # Fetch engagements for user
        user_engagements_for_content = fetch_data_stub().filter(
            Content.id == content_id
        ).all()

        content_df = pd.DataFrame(user_engagements_for_content)

        if len(content_df) == 0:
            return [], []

        content_embedding = team_wrappers[team].generate_content_embeddings(content_df)

        if len(content_embedding) == 0:
            return [], []

        return get_ANN_recommednations(content_embedding, team, K)
    except Exception as e:
        print(f"Error during get_ANN_recommendations_from_content recommendations for {team}: {e}")
        print(traceback.format_exc())
        return [], []
