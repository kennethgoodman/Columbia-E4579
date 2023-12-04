import numpy as np
from src.recommendation_system.recommendation_flow.ranking.AbstractRanker import AbstractRanker
from sklearn.preprocessing import StandardScaler

class CharlieRanker(AbstractRanker):
    def _rank_ids(self, user_id, content_ids, limit, probabilities, seed, starting_point, X=None):
        user_data = X[(X['user_id'] == user_id) & (X['content_id'].isin(content_ids))].dropna()
        user_data['like'], user_data['dislike'], user_data['engage_time'], user_data['content_id'] = probabilities
        # Calculate score

        # We are standardarize the engage time because the range is different as compared to like and dislike
        scaler = StandardScaler()
        user_data['engage_time'] = scaler.fit_transform(user_data['engage_time'].values.reshape(-1, 1))

        # Changing the range from [-1, 1] to [0, 1] for consitency
        user_data['engage_time'] = (user_data['engage_time'] + 1) / 2

        # We are combining like and dislike probability to capture the overall sentiment with more weights to dislike
        # Also including the engage time with weighted average with lower weight as "like" and "dislike" are stronger actions
        user_data['rank'] = 0.6 * (0.3 * user_data['like'] - 0.7 * user_data['dislike']) + 0.4 * user_data[
            'engage_time']

        ## Adding a column to represent the artist_style of the previous content
        user_data['prev_artist_style'] = user_data['artist_style'].shift(1)

        ## Creating a mask to identify rows where the current and previous artist_style are the same
        artist_style_mask = user_data['artist_style'] == user_data['prev_artist_style']

        ## Assigning a penalty to the rank for rows where the artist_style is the same as the previous one
        user_data.loc[artist_style_mask, 'rank'] -= 0.1

        ranked_pred = user_data.sort_values('rank', ascending=False)
        return ranked_pred['content_id'].tolist()[:limit]

    def _get_name(self):
        return "GolfRanker"
