import numpy as np
from src.recommendation_system.recommendation_flow.ranking.AbstractRanker import AbstractRanker


class FoxtrotRanker(AbstractRanker):
    def _rank_ids(self, user_id, content_ids, limit, probabilities, seed, starting_point, X=None):
        score_df = X[(X['user_id'] == user_id) & (X['content_id'].isin(content_ids))].dropna()
        score_df['like'], score_df['dislike'], score_df['engage_time'], score_df['content_id'] = probabilities
        # Calculate score
        def get_artist(content_id):
            content = score_df
            artist = content[content['content_id'] == content_id]['artist_style'].to_string()
            return artist[artist.rindex(' ', 0, ) + 1:]

        ranked_pred = score_df.sort_values('like', ascending=False)
        ranked_content_ids = ranked_pred['content_id'].tolist()
        swap_count = 1
        iterations = 0
        while swap_count > 0 and iterations < 15:
            swap_count = 0
            for i in list(range(len(ranked_content_ids)-1))[1:]:
                last_content_id = ranked_content_ids[i-1]
                last_artist = get_artist(last_content_id)
                this_content_id = ranked_content_ids[i]
                this_artist = get_artist(this_content_id)
                next_content_id = ranked_content_ids[i+1]
                next_artist = get_artist(next_content_id)
                if this_artist == last_artist:
                    ranked_content_ids[i] = next_content_id
                    ranked_content_ids[i+1] = this_content_id
                    swap_count += 1
            iterations += 1
        return ranked_content_ids

    def _get_name(self):
        return "FoxtrotRanker"
