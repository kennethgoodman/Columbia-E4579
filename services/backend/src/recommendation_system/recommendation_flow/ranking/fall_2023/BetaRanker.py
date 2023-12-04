import numpy as np
from src.recommendation_system.recommendation_flow.ranking.AbstractRanker import AbstractRanker
from sklearn.preprocessing import StandardScaler

class BetaRanker(AbstractRanker):
    def _rank_ids(self, user_id, content_ids, limit, probabilities, seed, starting_point, X=None):
        score_df = X[(X['user_id'] == user_id) & (X['content_id'].isin(content_ids))].dropna()
        score_df['rank_value'] = (10 * score_df['like']) - (5 * score_df['dislike']) - (
                    5 * score_df['engage_time'] / 380000)
        ranked_pred = score_df.sort_values('rank_value', ascending=False)

        # no two same artist_style next to each other
        def get_artist_style(content_id):
            return score_df[score_df['content_id'] == content_id].iloc[0]['artist_style']

        ranked_pred = ranked_pred['content_id'].tolist()
        hold_out = []
        res = [ranked_pred.pop(0)]
        for id in ranked_pred:
            if hold_out:
                if get_artist_style(hold_out[0]) != get_artist_style(res[-1]):
                    res.append(hold_out.pop(0))
            if get_artist_style(id) == get_artist_style(res[-1]):
                hold_out.append(id)
            else:
                res.append(id)
        return res

    def _get_name(self):
        return "BetaRanker"
