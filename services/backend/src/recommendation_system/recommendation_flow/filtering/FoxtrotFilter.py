from sqlalchemy.sql import text
from src import db
import pandas as pd
import numpy as np

class FoxtrotFilter(AbstractFilter):
    def filter_ids(self, content_ids, user_id):
        df_user_clusters_like = pd.read_csv(r"users_clusters2.csv", nrows=100)
        engagement_sql_statement = text(f"""SELECT * FROM engagement""")
        with db.engine.connect() as con:
            df_engagement = list(con.execute(engagement_sql_statement))
        df_cluster_dislike = df_engagement[
            (df_engagement["engagement_value"] == -1)
            & (df_engagement["engagement_type"] == "Like")
        ].merge(
            df_user_clusters_like[["user_id", "cluster_number"]],
            how="outer",
            on="user_id",
        )
        list_cluster_dislike = df_cluster_dislike.drop_duplicates(
            subset=["cluster_number", "content_id"], keep="first"
        )[["cluster_number", "content_id"]].values.tolist()
        list_cluster_dislike.sort(reverse=True)
        list_cluster = df_user_clusters_like.drop_duplicates(
            subset=["user_id", "cluster_number"], keep="first"
        )[["user_id", "cluster_number"]].values.tolist()
        list_content = content_ids
        list_to_filter = []
        list_filtered_disliked = []
        for id, cluster in list_cluster:
            if id == user_id:
                for cluster_dislike, content_id_dislike in list_cluster_dislike:
                    if cluster_dislike == cluster:
                        list_to_filter.append(content_id_dislike)
                list_filtered_disliked = [
                    x for x in list_content if x not in list_to_filter
                ]
        list_seen = (
            df_engagement[df_engagement["user_id"] == user_id]
            .drop_duplicates(subset="content_id", keep="first")["content_id"]
            .values.tolist()
        )
        filtered_content_ids = [x for x in list_filtered_disliked if x not in list_seen]
        return filtered_content_ids
