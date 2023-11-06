# -*- coding: utf-8 -*-
"""Charlie_Light_Filter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18sSfZiEfV2qrmi6tFHY9c5emK81MiANT

# Imports
"""

from src.recommendation_system.recommendation_flow.filtering.AbstractFilter import AbstractFilter
from src.recommendation_system.recommendation_flow.filtering.linear_model_helper import DataCollector


# from sqlalchemy.sql.schema import ScalarElementColumnDefault
# from typing import Tuple
# from google.colab import drive


"""# Your Implementation - Example Here, Must Modify"""

class DataCollectorCharlie(DataCollector):

	def coefficients(self):
    	return {
            'content_likes': -0.00023,
            'content_dislikes': 0.001026,
            'content_engagement_time_avg': 3.68948e-7,
            'user_likes': 4.275072e-6,
            'user_dislikes': -1.988219e-6,
            'user_engagement_time_avg': 3.185241e-7,
        }
	
  def artist_styles_one_hot(self):
      return [
            "medieval", "oil_on_canvas", "scifi", "leonardo_da_vinci", "movie: Batman", "movie: Gold"
        ], [
            -0.002887, -0.001683, -0.002155, 0.000314, 0.002131, 0.004903, -0.000623
        ]

  def sources_one_hot(self):
      return [
        "human_prompts", "r/EarthPorn", "r/SimplePrompts", "r/Cyberpunk", "r/whoahdude", "William Shakespeare"
    ], [
        0.000101, 0.00494, -0.001134, -0.000446, -0.001773, -0.000422, -0.001264
    ]

  def num_inference_steps_one_hot(self):
      return [
        100, 50
    ], [
        -0.00031, -0.000965, -0.000655
    ]

  def threshold(self):
      return 0.28

  def policy_filter_one(self, training_data, content_id):
      desired_styles = ['human_prompts', 'r/EarthPorn', 'r/Showerthoughts']
      artist_style = training_data[training_data['content_id'] == content_id]['artist_style'].values[0]
      if artist_style in desired_styles:
          return True
      else:
          return False

  def policy_filter_two(self, training_data, content_id):
      net_likes_threshold = 4
      training_data = training_data.merge(self.engagement_data[["content_id", "engagement_type", "engagement_value", ]],
                                          on="content_id", how="left")
      net_likes = \
      training_data[(training_data['content_id'] == content_id) & (training_data["engagement_type"] == "Like")][
          'engagement_value'].sum()
      if net_likes >= net_likes_threshold:
          return True
      else:
          return False

class CharlieFilter(AbstractFilter):
    def _filter_ids(self, user_id, content_ids, seed, starting_point):
        dc = DataCollectorCharlie()
        dc.gather_data(user_id, content_ids)
        dc.feature_eng()
        if starting_point.get("policy_filter_one", False):
            pf_one = set(
                content_id
                for content_id in content_ids
                if dc.policy_filter_one(dc.results, content_id)
            )
        else:
            pf_one = set(content_ids)
        if starting_point.get("policy_filter_two", False):
            pf_two = set(
                content_id
                for content_id in content_ids
                if dc.policy_filter_two(dc.results, content_id)
            )
        else:
            pf_two = set(content_ids)
        if starting_point.get("linear_model", False) and user_id not in [0, None]:
            pf_lr = set(dc.run_linear_model())
        else:
            pf_lr = set(content_ids)
        return set(pf_one) & set(pf_two) & set(pf_lr)

    def _get_name(self):
        return "CharlieFilter"


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    """# Example For Use In Production"""

    data_collector = DataCollectorCharlie()
    random_content_ids = pd.read_csv('sample_data/generated_content_metadata.csv', sep="\t")['content_id'].values
    data_collector.filter_content_ids(1, random_content_ids)

    """# Training"""

    data_collector = DataCollectorCharlie()
    data_collector.gather_training_data()
    training_data = data_collector.feature_eng_training()

    def get_Y(engagement_data: pd.DataFrame) -> pd.DataFrame:
        """Engineers target variable.
        Args
          data (pd.DataFrame): Engagement data.
        Returns
          pd.DataFrame: Dataframe of 3 columns; 'user_id', 'content_id', 'score',
            where 'score' being the target variable that you want to predict.
        """
        # Dummy target dataframe. Your output dataframe should have 3 columns; 'user_id', 'content_id', 'score'
        # Where 'score' being the target variable that you want to predict.
        engagement_data_copy = engagement_data.copy()
        weights = {"Like": 5, "MsecEngagedWith": 0.0005}
        engagement_data_copy["score"] = np.where(engagement_data_copy["engagement_type"] == "Like",
                                                 engagement_data_copy["engagement_value"] * weights["Like"],
                                                 engagement_data_copy["engagement_value"] * weights["MsecEngagedWith"])
        target_df = engagement_data_copy.groupby(["user_id", "content_id"])["score"].mean().to_frame().reset_index()
        s_scaler = StandardScaler()
        target_df["score"] = s_scaler.fit_transform(target_df[["score"]])

        # DO NOT CHANGE THIS. This step ensures that each row of the target variable (X)
        # corresponds to the correct row of features (y).
        target_df = pd.merge(
            training_data[['user_id', 'content_id']],
            target_df,
            on=['user_id', 'content_id'],
            how='left'
        )
        return target_df["score"]

    engagement_data = pd.read_csv('sample_data/engagement.csv', sep="\t")
    X = training_data[data_collector.get_columns()]
    y = get_Y(engagement_data)

    # training
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import accuracy_score

    # Split data into train and test: Add/change  other parametersas you wish
    # Also, feel free to use cross validation
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Depending on what your target variable y looks like, you have to choose a suitable model.
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"MSE: {np.mean((y_pred - y_test)**2)}")

    """# What You Need"""

    print("{")
    for x, y in zip(model.feature_names_in_, model.coef_):
      print(f"\t{x}: {y},")
    print("}")

    """# Policy Filtering 1"""

    def policy_filter_one(training_data, content_id):
        desired_styles = ['human_prompts', 'r/EarthPorn', 'r/Showerthoughts']
        artist_style = training_data[training_data['content_id'] == content_id]['artist_style'].values[0]
        if artist_style in desired_styles:
          return True
        else:
          return False


    policy_filter_one(
        training_data[training_data['content_id'] == random_content_ids[0]],
        random_content_ids[0]
    )

    """# Policy Filtering 2"""

    def policy_filter_two(training_data, content_id):
      filtered_content_ids = []
      net_likes_threshold = 4
      training_data = training_data.merge(engagement_data[["content_id", "engagement_type", "engagement_value", ]], on="content_id", how="left")
      net_likes = training_data[(training_data['content_id'] == content_id) & (training_data["engagement_type"] == "Like")]['engagement_value'].sum()
      if net_likes >= net_likes_threshold:
        return True
      else:
        return False


    policy_filter_two(
        training_data[training_data['content_id'] == random_content_ids[0]],
        random_content_ids[0]
    )

