
from sqlalchemy.sql.schema import ScalarElementColumnDefault
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

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
            self.generated_content_metadata_data[col_name] = self.generated_content_metadata_data[col_name].apply(lambda x: x if x in categories else 'other')
            encoder = OneHotEncoder(categories=[categories + ['other']], sparse=False)
            encoded_data = encoder.fit_transform(self.generated_content_metadata_data[[col_name]])
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
      df = pd.read_csv('engagement.csv', sep="\t")
      return df[df['content_id'].isin(content_ids)]

    def get_generated_content_metadata_data(self, content_ids):
      df = pd.read_csv('generated_content_metadata.csv', sep="\t")
      return df[df['content_id'].isin(content_ids)]

    def get_user_data(self, user_id):
      df = pd.read_csv('engagement.csv', sep="\t")
      return df[df['user_id'] == user_id]

    def gather_data(self, user_id, content_ids):
      self.engagement_data = self.get_engagement_data(content_ids)
      self.generated_content_metadata_data = self.get_generated_content_metadata_data(content_ids)
      self.user_data = self.get_user_data(user_id)

    def gather_training_data(self):
      self.engagement_data = pd.read_csv('engagement.csv', sep="\t")
      self.generated_content_metadata_data = pd.read_csv('generated_content_metadata.csv', sep="\t")
      self.user_data = pd.read_csv('engagement.csv', sep="\t")

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

      # content_results = pd.merge(
      #     generated_content_features,
      #     content_engagement_features,
      #     on='content_id',
      #     how='left'
      # ).fillna(0)

      # self.training_results = pd.merge(
      #     self.engagement_data,
      #     content_results,
      #     on='content_id',
      #     how='left',
      # ).fillna(0)

      # self.training_results = pd.merge(
      #     self.training_results,
      #     user_attr,
      #     on='user_id',
      #     how='left'
      # ).fillna(0)

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



data_collector = DataCollectorGolf()
filtered_content_ids = data_collector.get_filtered_content_ids()
print("Number of Filtered Content IDs:", len(filtered_content_ids))
print(filtered_content_ids)
