from src.recommendation_system.recommendation_flow.filtering.AbstractFilter import AbstractFilter
from src.recommendation_system.recommendation_flow.filtering.linear_model_helper import DataCollector


class DataCollectorEcho(DataCollector):
    def coefficients(self):
        return {
            'content_likes': 0.001327374532861648,
            'content_dislikes': -0.0025596352777526488,
            'content_engagement_time_avg': -1.1359359152350807e-10,
            'user_likes': 3.750439880704869e-05,
            'user_dislikes': -7.111467664335854e-06,
            'user_engagement_time_avg': -4.198888030205803e-09,
        }

    def artist_styles_one_hot(self):
        return [
            'anime', 'medieval', 'studio'
        ], [
            0.5, 0.5, 0.5, 0.5
        ]

    def sources_one_hot(self):
        return [
            'human_prompts', 'r/Showerthoughts', 'r/EarthPorn'
        ], [
            0.5, 0.5, 0.5, 0.5
        ]

    def num_inference_steps_one_hot(self):
        return [
            20, 50
        ], [
            0.5, 0.5, 0.5
        ]

    def threshold(self):
        return 300

    def coefficients(self):
        return {
            'content_likes': 104.48448080386746,
            'content_dislikes': -256.82015192805903,
            'content_engagement_time_avg': 8.057773633873186e-07,
            'user_likes': 0.7525136884076168,
            'user_dislikes': -0.1502183573794211,
            'user_engagement_time_avg': 0.00030911703068956127,
            'artist_style_anime': 52.408940915308825,
            'artist_style_medieval': -1.2635383495035055,
            'artist_style_studio': 56.92221997077624,
            'artist_style_other': -108.06762253650498,
            'source_human_prompts': 74.98897024194356,
            'source_r/Showerthoughts': -84.75159543701908,
            'source_r/EarthPorn': 103.55289816316545,
            'source_other': -93.79027296795866,
            'num_inference_steps_20': 5.577317053822645,
            'num_inference_steps_50': 38.993461681648775,
            'num_inference_steps_other': -44.57077873559393
        }

    def policy_filter_one(self, training_data, content_ids):
        df = training_data[training_data['content_id'].isin(content_ids)]
        all_content_ids = df['content_id']
        df_human = df[df['source'] == 'human_prompts']
        sorted_human_content = df_human.sort_values('content_dislikes', ascending=False)
        # Remove duplicates
        sorted_human_content = sorted_human_content.drop_duplicates(subset=['content_id'])
        bottom_60_percent_cutoff = int(len(sorted_human_content) * 0.75)
        bottom_content_ids = sorted_human_content.iloc[:bottom_60_percent_cutoff]['content_id']
        lst = list(set(all_content_ids) - set(bottom_content_ids))
        return lst

    def policy_filter_two(self, training_data, content_ids):
        df = training_data[training_data['content_id'].isin(content_ids)]
        all_content_ids = df['content_id']
        df_movie = df.dropna(subset=['artist_style'])
        df_movie = df_movie[df_movie['artist_style'].apply(lambda x: 'movie' in str(x).lower())]
        sorted_movie_content = df_movie.sort_values('content_dislikes', ascending=False)
        # Remove duplicates
        sorted_movie_content = sorted_movie_content.drop_duplicates(subset=['content_id'])
        bottom_60_percent_cutoff = int(len(sorted_movie_content) * 0.750)
        bottom_content_ids = sorted_movie_content.iloc[:bottom_60_percent_cutoff]['content_id']
        lst = list(set(all_content_ids) - set(bottom_content_ids))
        return lst


class EchoFilter(AbstractFilter):
    def _filter_ids(self, user_id, content_ids, seed, starting_point):
        dc = DataCollectorEcho()
        dc.gather_data(user_id, content_ids)
        dc.feature_eng()
        if starting_point.get("policy_filter_one", False):
            pf_one = dc.policy_filter_one(dc.results, content_ids)  # policy one used here
        else:
            pf_one = set(content_ids)
        if starting_point.get("policy_filter_two", False):
            pf_two = dc.policy_filter_two(dc.results, content_ids)  # policy two used here
        else:
            pf_two = set(content_ids)
        if starting_point.get("linear_model", False) and user_id not in [0, None]:
            pf_lr = set(dc.run_linear_model())
        else:
            pf_lr = set(content_ids)
        return pf_one & pf_two & pf_lr

    def _get_name(self):
        return "EchoFilter"
