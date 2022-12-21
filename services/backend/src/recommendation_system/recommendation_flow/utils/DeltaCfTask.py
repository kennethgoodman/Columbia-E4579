"""
Computing user/item embedding based on collaborative filtering
This script computes the collaborative embedding based on the user engagement
"""

from src.api.users.models import User
from src.api.users.crud import *
from src.api.content.models import Content, GeneratedContentMetadata
from src.api.engagement.models import Engagement, EngagementType
from src import create_app
from src import db

from sqlalchemy.sql import select
from sqlalchemy.orm import joinedload, load_only

from scipy.sparse import dok_matrix
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

import json

def generate_cf_embedding():
    # get all users
    users = get_all_users()
    num_users = len(users)

    # get all contents
    contents = Content.query.options(
            load_only(Content.id),
            joinedload(Content.generated_content_metadata).options(
                load_only(GeneratedContentMetadata.artist_style))).all()
    num_contents = len(contents)

    # get engagement
    engagements = Engagement().query.all()

    # retrieve all unique artist styles
    artist_styles = set()
    content_to_style = dict()
    stats = [0,0,0]
    for content in contents:
        artist_style = content.generated_content_metadata.artist_style

        if artist_style == "":
            stats[0] = stats[0] + 1
            content_to_style[content.id] = "Random"
        elif artist_style == "NA":
            stats[1] = stats[1] + 1
            content_to_style[content.id] = "Random"
        else:
            stats[2] = stats[2] + 1
            artist_styles.add(artist_style)
            content_to_style[content.id] = artist_style

    artist_styles.add("Random")

    # debug stats
    print(f"CF TASK: retrieved users: {num_users}, contents: {num_contents}")
    print(f"CF TASK: styles, empty ({stats[0]}), NA ({stats[1]}), found ({stats[2]})")

    # create the engagement matrix
    num_styles = len(artist_styles)
    cf_matrix_shape = (num_users, num_styles)
    engagement_matrix = dok_matrix(cf_matrix_shape)

    # create mappings: user.id -> i, style -> j, j -> style
    user_idx = dict()
    style_idx = dict()
    style_array = []

    for idx, user in enumerate(users):
        user_idx[user.id] = idx

    for idx, style in enumerate(artist_styles):
        style_idx[style] = idx
        style_array.append(style)

    # update the engagement matrix
    for engagement in engagements:

        if engagement.engagement_type == EngagementType.MillisecondsEngagedWith:
            # update the entry
            i = user_idx[engagement.user_id]
            j = style_idx[content_to_style[engagement.content_id]]

            engagement_matrix[i,j] = engagement_matrix[i,j] + engagement.engagement_value

    # create the factorization
    nmf = NMF(n_components = 2, init='random')
    U = nmf.fit_transform(normalize(engagement_matrix, norm='l2', axis = 1))
    V = nmf.components_

    # compute nearest neighbor for each user
    nn = NearestNeighbors(n_neighbors = 50, algorithm='ball_tree').fit(V.T)
    dists, indices = nn.kneighbors(U)

    # map back to categories for each user
    user_styles = []
    for index_list in indices:
        styles = []
        for idx in index_list:
            styles.append(style_array[idx])
        user_styles.append(styles)

    # create table to store prefs
    with db.engine.connect() as con:
        con.execute("CREATE TABLE IF NOT EXISTS user_prefs(id int, prefs JSON, primary key (id))")

    # insert the entries
    with db.engine.connect() as con:
        for idx, user in enumerate(users):
            prefs = json.dumps(user_styles[idx])
            con.execute(f"REPLACE INTO user_prefs (id, prefs) VALUES ({user.id},'{prefs}')")

if __name__ == '__main__':

    app = create_app()

    with app.app_context():
        generate_cf_embedding()
