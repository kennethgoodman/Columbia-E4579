import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.sql.expression import func
from sklearn.preprocessing import StandardScaler
from src.api.content.models import Content, GeneratedContentMetadata
from src.api.engagement.models import Engagement
from src import db


class UserBasedRecommender:

    _instance = None  # Singleton instance reference

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UserBasedRecommender, cls).__new__(cls)
            cls._instance.user_similarity_map = {}
            cls._instance.gather_data()
            cls._instance.compute_similarity()
        return cls._instance

    def gather_data(self):
        # Connect to the database and fetch user-content engagement.
        query = ("SELECT t_1.user_id AS user_id, num_of_contents, avg_engagement_time, num_of_likes, num_of_dislikes "
                 "FROM (SELECT user_id, COUNT(DISTINCT content_id) AS num_of_contents, "
                 "AVG(engagement_value) AS avg_engagement_time "
                 "FROM engagement WHERE engagement_type = \"MillisecondsEngagedWith\" AND engagement_value > 500 AND "
                 "engagement_value < 30000 "
                 "GROUP BY user_id) AS t_1 "
                 "JOIN (SELECT user_id, COUNT(*) AS num_of_likes FROM engagement "
                 "WHERE engagement_value = 1 GROUP BY user_id) AS t_2 ON t_1.user_id = t_2.user_id "
                 "JOIN (SELECT user_id, COUNT(*) AS num_of_dislikes FROM engagement WHERE engagement_value = -1 "
                 "GROUP BY user_id) AS t_3 ON t_2.user_id = t_3.user_id")
        with db.engine.connect() as con:
            result = con.execute(text(query))
        result_df = pd.DataFrame(result)
        data_columns = ["num_of_contents", "avg_engagement_time", "num_of_likes", "num_of_dislikes"]
        non_std_data = result_df[data_columns]
        scaler = StandardScaler()
        std_data = scaler.fit_transform(non_std_data)
        std_data = pd.DataFrame(std_data, columns=data_columns)
        result_df = pd.concat([result_df[["user_id"]], std_data], axis=1)
        return result_df

    def compute_similarity(self):
        data = self.gather_data()
        user_ids = data["user_id"].tolist()
        sim_matrix = pd.DataFrame(columns=data["user_id"], index=data["user_id"])
        data = data.set_index(keys="user_id")
        for i in user_ids:
            for j in user_ids:
                user_i = np.array(data.loc[i])
                user_j = np.array(data.loc[j])
                sim_matrix.loc[i, j] = user_i.dot(user_j) / (np.linalg.norm(user_i) * np.linalg.norm(user_j))
        for user in user_ids:
            self.user_similarity_map[user] = sim_matrix.loc[user].sort_values(ascending=False).index.tolist()[1:]

    def get_similar_users(self, user_id):
        # Fetch the list of similar users for a given user_id from the map.
        return self.user_similarity_map.get(user_id, list(self.user_similarity_map.keys()))

    def recommend_items(self, user_id, num_recommendations=10):
        similar_users = self.get_similar_users(user_id)
        scope = "(" + ", ".join(str(s_u_id) for s_u_id in similar_users) + ")"
        query_rec = ("SELECT content_id, engagement_value FROM engagement WHERE content_id NOT IN "
                     "(SELECT content_id FROM engagement WHERE user_id = " + str(user_id) + " OR "
                     "engagement_value = -1) AND user_id IN " + scope + " AND engagement_value < 30000 "
                     "ORDER BY content_id")
        with db.engine.connect() as con:
            result_rec = con.execute(text(query_rec))
        return list(result_rec)
