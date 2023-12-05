import numpy as np
from src.recommendation_system.recommendation_flow.ranking.AbstractRanker import AbstractRanker
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class AlphaRanker(AbstractRanker):
    def _rank_ids(self, user_id, content_ids, limit, probabilities, seed, starting_point, X=None):
        score_df = X[(X['user_id'] == user_id) & (X['content_id'].isin(content_ids))].dropna()
        score_df['like'], score_df['dislike'], score_df['engage_time'], score_df['content_id'] = probabilities
        scaler = MinMaxScaler()
        score_df['engage_time'] = scaler.fit_transform(score_df['engage_time'].values.reshape(-1, 1))
        score_df['artist_style_theme'] = score_df['artist_style'].apply(lambda x: 'movie' if 'movie' in str(x) else x)
        score_group_df = score_df.groupby(['artist_style_theme']).apply(
            lambda x: x.sort_values(['like', 'dislike'], ascending=False))
        score_group_df['like-dislike'] = score_group_df.apply(lambda x: x['like'] - x['dislike'], axis=1)
        score_group_df['overall_score'] = score_group_df.apply(
            lambda x: (0.8 * x['like-dislike']) + (0.2 * x['engage_time']) if x['like-dislike'] > 0 else (
                (0.6 * x['like-dislike']) + (0.4 * x['engage_time']) if x['like-dislike'] < 0 else x['engage_time']),
            axis=1)
        score_group_df = score_group_df.drop(columns={'artist_style_theme'})
        new_group_df = score_group_df.groupby(['artist_style_theme']).apply(
            lambda x: x.sort_values(['overall_score'], ascending=False))
        try:
            new_group_df = new_group_df.droplevel(0)
        except:
            pass
        new_group_df.reset_index(inplace=True)
        new_df = new_group_df[['content_id', 'artist_style_theme', 'overall_score']]
        output_df = new_df.sort_values(by=['overall_score', 'artist_style_theme'], ascending=[False, True])

        assorted_df = pd.DataFrame(columns=output_df.columns)
        style_counts = output_df.value_counts('artist_style_theme')

        for style, size in style_counts.items():
            if style == 'movie' and size / len(output_df) > 0.3:
                excl_movie_size = len(output_df) - size
                movie_diff = size / len(output_df) - 0.3
                style_counts[style] = size / excl_movie_size + ((size / excl_movie_size) * movie_diff)
            else:
                style_counts[style] = 0.3

        # Initialize an empty list to store the DataFrames
        dfs_to_concat = []

        # Iterate over each style
        for style, count in style_counts.items():
            # Filter the DataFrame for the current style and add it to the list
            dfs_to_concat.append(output_df[output_df['artist_style_theme'] == style])

        # Concatenate all the DataFrames in the list
        assorted_df = pd.concat(dfs_to_concat, ignore_index=True)

        assorted_df['tile'] = assorted_df.groupby(['artist_style_theme']).cumcount() + 1
        assorted_df = assorted_df.sort_values('tile')

        return assorted_df['content_id'].tolist()

    def _get_name(self):
        return "AlphaRanker"
