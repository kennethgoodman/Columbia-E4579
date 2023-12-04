from src.recommendation_system.recommendation_flow.filtering.AbstractFilter import AbstractFilter
from src.recommendation_system.recommendation_flow.filtering.linear_model_helper import AbstractFeatureEng
import pandas as pd

class FeatureEngGolf(AbstractFeatureEng):
    def artist_styles_one_hot(self):
        return [
            'van_gogh', 'jean-michel_basquiat'
        ], [
            0.5, 0.5, 0.5
        ]

    def sources_one_hot(self):
        return [
            'human_prompts', 'r/Showerthoughts'
        ], [
            0.5, 0.5, 0.5
        ]

    def num_inference_steps_one_hot(self):
        return [
            100
        ], [
            0.5, 0.5
        ]

    def one_hot_encoding_functions(self):
        return zip(
            [self.artist_styles_one_hot(), self.sources_one_hot(), self.num_inference_steps_one_hot()],
            ['artist_style', 'source', 'num_inference_steps']
        )

    def threshold(self):
        return 1.5

    def policy_filter_one(self, training_data, content_ids):
        df = training_data[training_data['content_id'].isin(content_ids)]
        all_content_ids = df['content_id']
        df = df.dropna(subset=['artist_style'])
        df_sorted = df.sort_values('user_likes', ascending=False)
        content_ids_out = df_sorted.iloc[int(len(df_sorted) * 0.9):]['content_id']
        return all_content_ids[~all_content_ids.isin(content_ids_out)].tolist()

    def policy_filter_two(self, training_data, content_ids):
        df = training_data[training_data['content_id'].isin(content_ids)]
        all_content_ids = df['content_id']
        df_sorted = df.sort_values('user_dislikes', ascending=True)
        content_ids_out = df_sorted.iloc[int(len(df_sorted) * 0.9):]['content_id']
        return all_content_ids[~all_content_ids.isin(content_ids_out)].tolist()

    def get_filtered_content_ids(self):
        training_data = self.get_training_data()
        filtered_content_ids = self.filter_with_regression(training_data)
        filtered_content_ids = self.policy_filter_one(training_data, filtered_content_ids)
        filtered_content_ids = self.policy_filter_two(training_data, filtered_content_ids)
        return filtered_content_ids


class GolfFilter(AbstractFilter):
    def _filter_ids(self, user_id, content_ids, seed, starting_point, amount=None, dc=None):
        golf_feature_eng = FeatureEngGolf(dc)
        golf_feature_eng.feature_eng()
        if starting_point.get("policy_filter_one", False):
            pf_one = golf_feature_eng.policy_filter_one(golf_feature_eng.results, content_ids)  # policy one used here
        else:
            pf_one = set(content_ids)
        if starting_point.get("policy_filter_two", False):
            pf_two = golf_feature_eng.policy_filter_two(golf_feature_eng.results, content_ids)  # policy two used here
        else:
            pf_two = set(content_ids)
        if starting_point.get("linear_model", False) and user_id not in [0, None]:
            pf_lr = set(golf_feature_eng.run_linear_model())
        else:
            pf_lr = set(content_ids)
        return set(pf_one) & set(pf_two) & set(pf_lr)

    def _get_name(self):
        return "GolfFilter"
