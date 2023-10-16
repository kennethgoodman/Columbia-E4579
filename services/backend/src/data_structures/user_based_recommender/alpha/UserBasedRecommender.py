from sqlalchemy.sql.expression import func
from src.api.content.models import Content, GeneratedContentMetadata
from src.api.engagement.models import Engagement
from src import db
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedRecommender:
    _instance = None  # Singleton instance reference

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UserBasedRecommender, cls).__new__(cls)
            cls._instance.user_content_matrix = {}
            cls._instance.user_similarity_map = {}
            cls._instance.gather_data()
            cls._instance.compute_similarity()
        return cls._instance

    # Create a mapping function
    def map_engagement_value(self, row, factor):
        if str(row["engagement_type"]) == "EngagementType.Like":
            return row["engagement_value"] * 1000
        elif str(row["engagement_type"]) == "EngagementType.MillisecondsEngagedWith":
            return row["engagement_value"] / factor
        else:
            return None  # Handle other cases if needed

    def gather_data(self):
        # Connect to the database and fetch user-content engagement.
        self.interactions = db.session.query(
            Engagement.user_id,
            Engagement.content_id,
            Engagement.engagement_type,
            Engagement.engagement_value,
        ).all()

        df = pd.DataFrame(
            self.interactions,
            columns=["user_id", "content_id", "engagement_type", "engagement_value"],
        )
        engagement_time = df.loc[df["engagement_value"] >= 500, ["engagement_value"]]
        q99 = float(engagement_time.quantile(0.99))

        # Preprocessing
        # Dont forget to transform db.session to dataframe
        df["engagement_value"] = df.apply(self.map_engagement_value, factor=q99, axis=1)
        impression = df[["user_id", "content_id", "engagement_value"]]
        agg_impression = (
            impression.groupby(["user_id", "content_id"])
            .agg({"engagement_value": "sum"})
            .reset_index()
        )

        for user_id, content_id, value in agg_impression.values:
            if user_id not in self.user_content_matrix:
                self.user_content_matrix[user_id] = {}
            self.user_content_matrix[user_id][content_id] = value

    def compute_similarity(self):
        # Compute the similarity between users.
        # For simplicity, we'll use cosine similarity as our metric.

        # Convert our dict to a matrix for computation
        users = list(self.user_content_matrix.keys())
        content_items = set(
            item for sublist in self.user_content_matrix.values() for item in sublist
        )
        matrix = []

        for user in users:
            row = [
                self.user_content_matrix[user].get(content, 0)
                for content in content_items
            ]
            matrix.append(row)

        # compute pairwise similarity
        # input: user-content engagement matrix; rows represent users and columns represent contents
        # output: user-user pairwise similarities; rows represents users and columns also represent users
        similarities = cosine_similarity(matrix)

        # Update self.user_similarity_map with the similarities.
        for idx, user in enumerate(users):
            self.user_similarity_map[user] = [
                (users[i], sim) for i, sim in enumerate(similarities[idx]) if i != idx
            ]

    def get_similar_users(self, user_id):
        # Fetch the list of similar users for a given user_id from the map.
        return sorted(
            self.user_similarity_map.get(user_id, []), key=lambda x: x[1], reverse=True
        )

    def recommend_items(self, user_id, num_recommendations=10):
        # For a given user, fetch the list of similar users.
        # Experimenting: Just pick top 5 similar users
        similar_users = self.get_similar_users(user_id)[:3]
        recommended = set()

        # Recommend items engaged by those users, which the given user hasn't seen.
        # Don't forget to just add the content id if the engagement value > 0

        for sim_user, _ in similar_users:
            for content_id, value in self.user_content_matrix.get(sim_user, {}).items():
                recommended.add((content_id, value))

        recommended_tuple = sorted(recommended, key=lambda x: x[1], reverse=True)
        recommended_content_ids = [int(item[0]) for item in recommended_tuple]
        content_ids_value = [int(item[1]) for item in recommended_tuple]

        # Return only the top num_recommendations
        # changing for 900 for now to compensate 2tower model
        return (
            recommended_content_ids[:900],
            content_ids_value[:900],
        )
