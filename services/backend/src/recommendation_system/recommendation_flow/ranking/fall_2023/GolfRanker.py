import numpy as np
from src.recommendation_system.recommendation_flow.ranking.AbstractRanker import AbstractRanker


class GolfRanker(AbstractRanker):
    def _rank_ids(self, user_id, content_ids, limit, probabilities, seed, starting_point, X=None):
        user_df = X[(X['user_id'] == user_id) & (X['content_id'].isin(content_ids))].dropna()
        # Calculate score
        weights = {}
        weights['like'] = user_df['user_numerical_feature'].mean() / 100
        weights['dislike'] = -weights['like'] / 2
        weights['engage_time'] = user_df['user_time_elapsed'].mean() / 100

        user_df['like'], user_df['dislike'], user_df['engage_time'], content_ids = probabilities
        user_df['score'] = (
                user_df['like'] * weights['like'] +
                user_df['dislike'] * weights['dislike'] +
                user_df['engage_time'] * weights['engage_time']
        )
        user_df['content_id'] = content_ids
        # Rank
        ranked_df = user_df.sort_values('score', ascending=False)
        return ranked_df['content_id'].tolist()

    def _get_name(self):
        return "GolfRanker"
