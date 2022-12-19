#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import Python packages
from sqlalchemy.sql import text
from src import db
import pandas as pd
import numpy as np

class FoxtrotFiltering:
    def filter_ids(content_ids, df_user_clusters_like, user_id):
        # get data from mysql
        engagement_sql_statement = text(f"""SELECT * FROM engagement""")
        with db.engine.connect() as con:
            df_engagement = list(con.execute(engagement_sql_statement))
        # create dataframe of liked images
        df_cluster_dislike = df_engagement[(df_engagement['engagement_value'] == -1) & (df_engagement['engagement_type'] == 'Like')].merge(df_user_clusters_like[['user_id', 'cluster_number']], how='outer', on='user_id')
        # create list of disliked images by cluster
        list_cluster_dislike = df_cluster_dislike.drop_duplicates(subset=["cluster_number", "content_id"], keep='first')[['cluster_number', 'content_id']].values.tolist()
        list_cluster_dislike.sort(reverse=True)
        # create list of user-cluster
        list_cluster = df_user_clusters_like.drop_duplicates(subset=["user_id", "cluster_number"], keep='first')[['user_id', 'cluster_number']].values.tolist()
        list_content = content_ids
        list_to_filter = []
        list_filtered_disliked = []
        # get the filtered content by dislike
        for id, cluster in list_cluster:
            if id == user_id:
                for cluster_dislike, content_id_dislike in list_cluster_dislike:
                    if cluster_dislike == cluster:
                        list_to_filter.append(content_id_dislike)
                list_filtered_disliked = [x for x in list_content if x not in list_to_filter]
        # filter content by seen images
        list_seen = df_engagement[df_engagement['user_id'] == user_id].drop_duplicates(subset="content_id", keep='first')['content_id'].values.tolist()
        filtered_content_ids = [x for x in list_filtered_disliked if x not in list_seen]
        return filtered_content_ids

