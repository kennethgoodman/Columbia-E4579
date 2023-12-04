import numpy as np
from src.recommendation_system.recommendation_flow.ranking.AbstractRanker import AbstractRanker
from sklearn.preprocessing import StandardScaler
import pandas as pd

class DeltaRanker(AbstractRanker):
    def _rank_ids(self, user_id, content_ids, limit, probabilities, seed, starting_point, X=None):
        user_df = X[(X['user_id'] == user_id) & (X['content_id'].isin(content_ids))].dropna()
        user_df['like'], user_df['dislike'], user_df['engage_time'], user_df['content_id'] = probabilities
        def select_artist_style(style):
            if pd.isna(style) or str(style).startswith('movie:'):
                return 'other'
            else:
                return style

        user_df['selected_artiststyle'] = user_df['artist_style'].apply(select_artist_style)

        # value function formula: like - dislike + int(engagetime/1000) (1 point for every 4 seconds)
        user_df['value'] = user_df['like'] - user_df['dislike'] + (user_df['engage_time'] / 4000).astype(int)
        user_df_sorted = user_df.sort_values(by='value', ascending=False)

        # additional ordering: if not have specific art styles, order so no same styles seen consequently
        sorted_content_ids = []
        last_artist_style = 1
        keep_styles = {'other', 'gta_v', 'medieval', 'detailed_portrait', 'van_gogh', 'unreal_engine', 'face_and_lighting', 'scifi', 'oil_on_canvas', 'anime', 'studio'}

        while not user_df_sorted.empty:
            selected_rows = user_df_sorted.loc[(user_df_sorted['selected_artiststyle'] != last_artist_style) | (user_df_sorted['selected_artiststyle'].isin(keep_styles))]
            if selected_rows.empty:
                break
            selected_row = selected_rows.iloc[0]
            sorted_content_ids.append(selected_row['content_id'])
            last_artist_style = selected_row['selected_artiststyle']
            user_df_sorted = user_df_sorted.drop(selected_row.name)
        return sorted_content_ids

    def _get_name(self):
        return "DeltaRanker"
