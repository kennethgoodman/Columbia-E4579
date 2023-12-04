import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class AbstractFeatureEng:
    def __init__(self, dc):
        self.dc = dc

    def artist_styles_one_hot(self):
        raise NotImplementedError(
            "you need to implement this, needs to be two lists, one for string one for coefficient, coefficient list "
            "is one larger to account for 'other'"
        )

    def sources_one_hot(self):
        raise NotImplementedError(
            "you need to implement this, needs to be two lists, one for string one for coefficient, coefficient list "
            "is one larger to account for 'other'"
        )

    def num_inference_steps_one_hot(self):
        raise NotImplementedError(
            "you need to implement this, needs to be two lists, one for string one for coefficient, coefficient list "
            "is one larger to account for 'other'"
        )

    def one_hot_encoding_functions(self):
        return zip(
            [self.artist_styles_one_hot(), self.sources_one_hot(), self.num_inference_steps_one_hot()],
            ['artist_style', 'source', 'num_inference_steps']
        )

    def custom_aggregation(self, prefix, data):
        result = {
            f'{prefix}_likes': np.sum((data['engagement_type'] == 'Like') & (data['engagement_value'] == 1)),
            f'{prefix}_dislikes': np.sum((data['engagement_type'] == 'Like') & (data['engagement_value'] == -1)),
            f'{prefix}_engagement_time_avg': data[data['engagement_type'] == 'MillisecondsEngagedWith'][
                'engagement_value'].mean(),
        }
        return pd.Series(result)

    def feature_generation_user(self):
        # Use `groupby` without including 'user_id' in the Series returned by `custom_aggregation`
        aggregated_data = self.user_data.groupby('user_id').apply(lambda data: self.custom_aggregation('user', data))

        # Drop the 'user_id' level from the index if it's there to avoid conflicts when resetting the index
        if 'user_id' in aggregated_data:
            aggregated_data = aggregated_data.drop('user_id', axis=1)

        # Reset the index after ensuring 'user_id' is not in the columns
        return aggregated_data.reset_index()

    def feature_generation_content_one_hot_encoding(self):
        for (categories, _coefficient), col_name in self.one_hot_encoding_functions():
            transformed_col = self.generated_content_metadata_data[col_name].apply(
                lambda x: x if x in categories else 'other').to_frame()
            encoder = OneHotEncoder(categories=[categories + ['other']], sparse=False)
            encoded_data = encoder.fit_transform(transformed_col)
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col_name]))
            for col in encoded_df.columns:
                self.generated_content_metadata_data[col] = encoded_df[col]
        return self.generated_content_metadata_data

    def feature_generation_content_engagement_value(self):
        return self.engagement_data.groupby('content_id').apply(
            lambda data: self.custom_aggregation('content', data)
        ).reset_index()

    def feature_generation(self):
        self.feature_generation_user()
        self.feature_generation_content_one_hot_encoding()
        self.feature_generation_content_engagement_value()

    def feature_eng(self):
        user_attr = self.feature_generation_user()
        if len(user_attr) == 0:
            self.results = pd.DataFrame()
            return self.results
        content_engagement_features = self.feature_generation_content_engagement_value()
        generated_content_features = self.feature_generation_content_one_hot_encoding()
        interaction_pairs = self.engagement_data[
            ['user_id', 'content_id']].drop_duplicates()
        self.results = pd.merge(
            interaction_pairs,
            user_attr,
            on='user_id',
            how='left'
        ).fillna(0)
        content_results = pd.merge(
            generated_content_features,
            content_engagement_features,
            on='content_id',
            how='left'
        ).fillna(0)
        self.results = pd.merge(
            self.results,
            content_results,
            on='content_id',
            how='left'
        ).fillna(0)
        return self.results

    def threshold(self):
        raise NotImplementedError("you need to implement")

    def coefficients(self):
        return {
            'content_likes': 0.0,
            'content_dislikes': 0.0,
            'content_engagement_time_avg': 0.0,

            'user_likes': 0.0,
            'user_dislikes': 0.0,
            'user_engagement_time_avg': 0.0,
        }

    def run_linear_model(self, content_ids_to_run_one=None):
        coeffs = self.coefficients()
        for (categories, _coefficients), col_name in self.one_hot_encoding_functions():
            for category, coefficient in zip(categories + ['other'], _coefficients):
                if col_name + "_" + str(category) not in coeffs:
                    coeffs[col_name + "_" + str(category)] = 0
                coeffs[col_name + "_" + str(category)] += coefficient

        df_to_run_on = self.results.copy()
        if content_ids_to_run_one is not None:
            df_to_run_on = df_to_run_on[df_to_run_on['content_id'].isin(content_ids_to_run_one)]
        df_to_run_on['linear_output'] = 0.0
        for col_name, _coefficient in coeffs.items():
            df_to_run_on['linear_output'] += df_to_run_on[col_name] * _coefficient
        return df_to_run_on[df_to_run_on['linear_output'] >= self.threshold()]['content_id'].values

    def filter_content_ids(self, dc, content_ids):
        self.engagement_data, self.generated_content_metadata_data, self.user_data = dc.return_data_copy()
        self.feature_eng()
        return self.run_linear_model()
