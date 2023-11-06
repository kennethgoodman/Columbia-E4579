import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sqlalchemy.sql.schema import ScalarElementColumnDefault
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class DataCollector:

    def artist_styles_one_hot(self):
        raise NotImplementedError(
            "you need to implement this, needs to be two lists, one for string one for coefficient, coefficient list is one larger to account for 'other'"
            "Coefficient is from the model after training, so to prepare training data, you can put dummy number first, then replace it later after model has been trained"
        )

    def sources_one_hot(self):
        raise NotImplementedError(
            "you need to implement this, needs to be two lists, one for string one for coefficient, coefficient list is one larger to account for 'other'"
            "Coefficient is from the model after training, so to prepare training data, you can put dummy number first, then replace it later after model has been trained"
        )

    def num_inference_steps_one_hot(self):
        raise NotImplementedError(
            "you need to implement this, needs to be two lists, one for string one for coefficient, coefficient list is one larger to account for 'other'"
            "Coefficient is from the model after training, so to prepare training data, you can put dummy number first, then replace it later after model has been trained"
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
            f'{prefix}_engagement_time_avg': data[data['engagement_type'] == 'MillisecondsEngagedWith']['engagement_value'].mean(),
        }
        return pd.Series(result)

    def feature_generation_user(self):
        return self.user_data.groupby('user_id').apply(lambda data: self.custom_aggregation('user', data)).reset_index()

    def feature_generation_content_one_hot_encoding(self):
        for (categories, _coefficient), col_name in self.one_hot_encoding_functions():
            transformed_col = self.generated_content_metadata_data[col_name].apply(lambda x: x if x in categories else 'other').to_frame()
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

    def get_engagement_data(self, content_ids):
      df = pd.read_csv('sample_data/engagement.csv', sep="\t")
      return df[df['content_id'].isin(content_ids)]

    def get_generated_content_metadata_data(self, content_ids):
      df = pd.read_csv('sample_data/generated_content_metadata.csv', sep="\t")
      return df[df['content_id'].isin(content_ids)]

    def get_user_data(self, user_id):
      df = pd.read_csv('sample_data/engagement.csv', sep="\t")
      return df[df['user_id'] == user_id]

    def gather_data(self, user_id, content_ids):
      self.engagement_data = self.get_engagement_data(content_ids)
      self.generated_content_metadata_data = self.get_generated_content_metadata_data(content_ids)
      self.user_data = self.get_user_data(user_id)

    def gather_training_data(self):
      self.engagement_data = pd.read_csv('sample_data/engagement.csv', sep="\t")
      self.generated_content_metadata_data = pd.read_csv('sample_data/generated_content_metadata.csv', sep="\t")
      self.user_data = pd.read_csv('sample_data/engagement.csv', sep="\t")

    def feature_eng_training(self):
      user_attr = self.feature_generation_user()
      content_engagement_features = self.feature_generation_content_engagement_value()
      generated_content_features = self.feature_generation_content_one_hot_encoding()

      interaction_pairs = self.engagement_data[
          ['user_id', 'content_id']].drop_duplicates()

      self.training_results = pd.merge(
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

      self.training_results = pd.merge(
          self.training_results,
          content_results,
          on='content_id',
          how='left'
      ).fillna(0)

      return self.training_results

    def feature_eng(self):
      user_attr = self.feature_generation_user()
      content_engagement_features = self.feature_generation_content_engagement_value()
      generated_content_features = self.feature_generation_content_one_hot_encoding()
      self.results = pd.merge(
          generated_content_features,
          content_engagement_features,
          on='content_id',
          how='left'
      ).fillna(0)
      self.results['user_id'] = user_attr['user_id'].iloc[0]
      self.results = pd.merge(
          self.results,
          user_attr,
          on='user_id'
      )

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

    def get_columns(self):
      cols = list(self.coefficients().keys())
      for (categories, _coefficients), col_name in self.one_hot_encoding_functions():
          for category, coefficient in zip(categories + ['other'], _coefficients):
            cols.append(col_name + "_" + str(category))
      return cols

    def run_linear_model(self):
        coeffs = self.coefficients()
        for (categories, _coefficients), col_name in self.one_hot_encoding_functions():
          for category, coefficient in zip(categories + ['other'], _coefficients):
            coeffs[col_name + "_" + str(category)] = coefficient

        self.results['linear_output'] = 0.0
        for col_name, _coefficient in coeffs.items():
            self.results['linear_output'] += self.results[col_name] * _coefficient
        return self.results[self.results['linear_output'] >= self.threshold()]['content_id'].values

    def filter_content_ids(self, user_id, content_ids):
      self.gather_data(user_id, content_ids)
      self.feature_eng()
      return self.run_linear_model()

class DataCollectorFoxtrot(DataCollector):
    def coefficients(self):
        return {
            'content_likes': 0.001327374532861648,
            'content_dislikes': -0.0025596352777526488,
            'content_engagement_time_avg': -1.1359359152350807e-10,

            'user_likes': 3.750439880704869e-05,
            'user_dislikes': -7.111467664335854e-06,
            'user_engagement_time_avg':  -4.198888030205803e-09,
        }

    def artist_styles_one_hot(self):
        return ['medieval','anime','studio','oil_on_canvas','unreal_engine', 'edward_hopper','shepard_fairey',], [
                0.00297867123567935, -0.00013583832416803058, 0.0011875140593540518, -0.0008508512202171477,
                -0.0011910319715602647, -0.0009473978012908358, 0.0015827575817499537, -0.002623823559547084]

    def sources_one_hot(self):
        return ['human_prompts','r/EarthPorn','r/Showerthoughts','r/scifi','r/pics','r/educationalgifs','r/Damnthatsinteresting',], [
                0.003435938296346102, 0.002055277050584419, -0.0016261067852378207, -1.3188188316680503e-05, -0.0014536933603578126,
                -0.000850669870952967, -0.0013183930372200371, -0.00022916410484521567]

    def num_inference_steps_one_hot(self):
        return [20, 50, 100], [0.0003241249776025939, 0.00013478447734715756, 0.0015154732174153412, -0.0019743826723650913]


    def threshold(self):
        return 0.005


    def policy_filter_one(self):
        temp = self.engagement_data.loc[self.engagement_data["engagement_type"] == "Like"].groupby("content_id")['engagement_value'].sum().rename('score', inplace=True).to_frame()
        temp = temp.loc[temp['score'] > 0]
        self.results = self.results[self.results['content_id'].isin(temp.index)]
        self.results = self.results.reset_index(drop=True)

    def policy_filter_two(self):
        temp2 = self.engagement_data.loc[self.engagement_data[
            'engagement_type']=='MillisecondsEngagedWith'].groupby(
            'content_id')['engagement_value'].count().rename(
            'count',inplace=True).to_frame().sort_values(by='count').iloc[50:]
        self.results = self.results[self.results['content_id'].isin(temp2.index)].reset_index(drop=True)

    def filter_content_ids(self, user_id, content_ids):
        self.gather_data(user_id, content_ids)
        self.feature_eng()
        self.policy_filter_one() # policy one used here
        self.policy_filter_two() # policy two used here
        return self.run_linear_model()
