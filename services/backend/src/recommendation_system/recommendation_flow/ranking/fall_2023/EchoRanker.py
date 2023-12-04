import numpy as np
from src.recommendation_system.recommendation_flow.ranking.AbstractRanker import AbstractRanker


class EchoRanker(AbstractRanker):
    def _rank_ids(self, user_id, content_ids, limit, probabilities, seed, starting_point, X=None):
        """Ranks the items for a given user based on your own criteria.

        Args:
            score_df (pd.DataFrame): Predicted-score Dataframe of columns;
                'user_id', 'content_id', 'like', 'dislike', 'engage_time', and
                also columns for content metadata.
            user_id (int): User ID to rank the items for.
            content_ids (Optional[list]): List of content ids to be considered for ranking.
        """
        user_df = score_df = X[(X['user_id'] == user_id) & (X['content_id'].isin(content_ids))].dropna()
        user_df['like'], user_df['dislike'], user_df['engage_time'], user_df['content_id'] = probabilities
        # Standardize the engagement time (z-score normalization)
        mean_engage_time = user_df['engage_time'].mean()
        std_engage_time = user_df['engage_time'].std()
        user_df['standardized_engage_time'] = (user_df['engage_time'] - mean_engage_time) / std_engage_time

        # Cap the standardized values at -2 and 2, then normalize to range 0 to 0.5
        user_df['standardized_engage_time'] = user_df['standardized_engage_time'].clip(lower=-2, upper=2)
        user_df['standardized_engage_time'] = (user_df['standardized_engage_time'] + 2) / 8

        # Calculate a composite score considering likes, dislikes, and capped standardized engagement time
        user_df['composite_score'] = user_df['like'] - user_df['dislike'] - user_df['standardized_engage_time']

        # Sort by composite score in descending order
        ranked_pred = user_df.sort_values('composite_score', ascending=False)
        from collections import deque
        duplicate_ids = deque()
        pred_content_ids = ranked_pred['content_id'].tolist()

        result_ids = [pred_content_ids[0]]

        for i in range(1, len(pred_content_ids)):
            while duplicate_ids:
                current_style = score_df.loc[score_df['content_id'] == duplicate_ids[0], "artist_style"].iloc[0]
                prev_style = score_df.loc[score_df['content_id'] == result_ids[-1], "artist_style"].iloc[0]
                if current_style != prev_style:
                    id = duplicate_ids.popleft()
                    result_ids.append(id)
                else:
                    break
            current_id = pred_content_ids[i]
            current_style = score_df.loc[score_df['content_id'] == current_id, "artist_style"].iloc[0]
            prev_style = score_df.loc[score_df['content_id'] == result_ids[-1], "artist_style"].iloc[0]
            if current_style == prev_style:
                duplicate_ids.append(current_id)
                continue
            result_ids.append(current_id)
        return result_ids[:limit]

    def _get_name(self):
        return "EchoRanker"
