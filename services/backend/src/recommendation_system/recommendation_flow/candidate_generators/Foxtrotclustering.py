import pandas as pd
import csv

import numpy as np
from sklearn.cluster import KMeans

# some auxiliary code we used to generate the csv with users clusters:
class FGenerator:

    # cluster all images from prompt embeddings
    def cluster_items_from_embeddings(self, number_rows, nb_clusters):

        # read images data
        df = pd.read_csv(r"generated_content_metadata.csv", nrows=number_rows)

        # put embeddings in right format
        x = pd.DataFrame(df["prompt_embedding"])
        x["embedding"] = x.apply(generator.convert_to_list, axis=1)
        df3 = pd.DataFrame(x["embedding"].to_list())

        # cluster data
        kmeans = KMeans(nb_clusters)
        kmeans.fit(df3)
        identified_clusters = kmeans.fit_predict(df3)
        print(identified_clusters)

        # insert additional column with value equal to cluster number
        df.insert(loc=0, column="clusters", value=identified_clusters)

        return df

    def cluster_users(self):

        # nb of users and items clusters
        nb_clusters_items = 10
        nb_clusters_users = 6

        row_items_offset = 150000
        row_users_offset = 40000

        df_items = self.cluster_items_from_embeddings(
            row_items_offset, nb_clusters_items
        )

        # print(df_items['content_id'])

        users_df = pd.read_csv(r"engagement.csv", nrows=row_users_offset)
        users_df = users_df.loc[users_df["engagement_type"] == "Like"]

        nb_users = users_df["user_id"].nunique()
        users_id = pd.unique(users_df["user_id"])
        # print(nb_users, users_id)

        df_liked = users_df.loc[users_df["engagement_value"] == 1]
        df_disliked = users_df.loc[users_df["engagement_value"] == -1]

        df_user_clusters_like = pd.DataFrame(users_id, columns=["user_id"])

        # for each of the items cluster, get the proportion of total images liked/disliked that come from this cluster
        df_user_clusters_like["clusters_portion_liked"] = df_user_clusters_like[
            "user_id"
        ].apply(
            lambda x: self.get_cluster_liked_per_user(
                df_liked, df_items, x, nb_clusters_items
            )
        )
        df_user_clusters_like["clusters_portion_disliked"] = df_user_clusters_like[
            "user_id"
        ].apply(
            lambda x: self.get_cluster_liked_per_user(
                df_disliked, df_items, x, nb_clusters_items
            )
        )
        # print(df_user_clusters_like)

        # cluster users based on similarities in proportions of likes for each cluster
        df3 = pd.DataFrame(df_user_clusters_like["clusters_portion_liked"].to_list())
        # df4 = pd.DataFrame(df_user_clusters_like['clusters_portion_disliked'].to_list())
        # df3 = pd.concat([df3, df4])

        kmeans = KMeans(nb_clusters_users)
        kmeans.fit(df3)
        identified_clusters = kmeans.fit_predict(df3)
        # print(identified_clusters)

        df_user_clusters_like["cluster_number"] = identified_clusters
        # print(df_user_clusters_like)

        # write resulting file in csv format
        df_user_clusters_like.to_csv("users_clusters2.csv")

        return None

    # return list with fraction of clusters liked (or disliked) by user
    def get_cluster_liked_per_user(self, df, df_items, user, nb_clusters_items):
        df_liked = df.loc[df["user_id"] == user]
        df_liked = df_liked["content_id"]

        # replace the content_ids by their cluster number
        df2 = df_liked.apply(lambda x: self.get_cluster_from_content_id(df_items, x))
        # print(df2)

        # add 1 row if cluster never liked
        df2 = pd.DataFrame(df2)
        for i in range(nb_clusters_items):
            if i not in df2["content_id"].values:
                tempDf = pd.DataFrame(columns=["content_id"])
                tempDf["content_id"] = i
                df2 = pd.concat([df2, tempDf])

        df2 = df2["content_id"]
        # return the fractions of images from total likes that come from the cluster
        list_clusters = df2.value_counts(normalize=True).sort_index().to_list()

        if len(list_clusters) <= nb_clusters_items:
            for i in range(nb_clusters_items - len(list_clusters)):
                list_clusters.append(0.0)

        return list_clusters

    # get the cluster number corresponding to a given content_id
    def get_cluster_from_content_id(self, df_items, content):
        # print('here', content)
        cluster = df_items.loc[df_items["content_id"] == content]["clusters"].iloc[0]
        return cluster

    def convert_to_list(self, row):
        str = row[0]
        str = str.replace("[", "")
        str = str.replace("]", "")
        li = list(str.split(","))
        li = [float(a) for a in li]

        return li

    # def retrieve_cluster(self, user):
    # 	df = pd.read_csv(r'users_clusters2.csv', nrows=100)
    # 	cluster_nb = df.loc[df['user_id'] == user]['cluster_number'].iloc[0]
    # 	users_in_cluster = df.loc[df['cluster_number'] == cluster_nb]['user_id'].to_list()
    # 	return users_in_cluster


if __name__ == "__main__":
    generator = FGenerator()
    # y = generator.cluster_users()
