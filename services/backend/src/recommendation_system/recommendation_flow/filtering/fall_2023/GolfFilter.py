from src.recommendation_system.recommendation_flow.filtering.AbstractFilter import AbstractFilter
from src.recommendation_system.recommendation_flow.filtering.linear_model_helper import DataCollector


class DataCollectorGolf(DataCollector):
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

    def get_training_data(self):
        self.gather_training_data()
        training_data = self.feature_eng_training()
        return training_data

    def get_Y(self, engagement_data):
        import pandas as pd
        target_df = engagement_data.groupby(
            ['user_id', 'content_id']
        )['engagement_value'].sum().rename('score', inplace=True).to_frame().reset_index()

        target_df = pd.merge(
            self.training_results[['user_id', 'content_id']],
            target_df,
            on=['user_id', 'content_id'],
            how='left'
        )

        return target_df['score']

    def filter_with_regression(self, training_data):
        from sqlalchemy.sql.schema import ScalarElementColumnDefault
        import pandas as pd
        from sklearn.preprocessing import OneHotEncoder
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import accuracy_score
        engagement_data = pd.read_csv('engagement.csv', sep="\t")
        X = training_data[self.get_columns()]
        y = self.get_Y(engagement_data)

        random_seed = 45
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

        model = LinearRegression()
        model.fit(X_train, y_train)

        content_ids = training_data['content_id']

        y_predict_whole = model.predict(X)

        errors = np.abs(y - y_predict_whole)
        best_cutoff = 0
        best_mse = float('inf')

        for cutoff in np.arange(20000, 22000, 10):
            mse = ((y[errors <= cutoff] - y_predict_whole[errors <= cutoff]) ** 2).mean()
            if mse < best_mse:
                best_mse = mse
                best_cutoff = cutoff

        selected_content_ids = content_ids[errors <= best_cutoff]

        return selected_content_ids

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
    def _filter_ids(self, user_id, content_ids, seed, starting_point):
        dc = DataCollectorGolf()
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
        return set(pf_one) & set(pf_two) & set(pf_lr)

    def _get_name(self):
        return "GolfFilter"
