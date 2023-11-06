# -*- coding: utf-8 -*-

from src.recommendation_system.recommendation_flow.filtering.AbstractFilter import AbstractFilter
from src.recommendation_system.recommendation_flow.filtering.linear_model_helper import DataCollector
import pandas as pd
import numpy as np

class DataCollectorBeta(DataCollector):
    def artist_styles_one_hot(self):
        return ['van_gogh', 'jean-michel_basquiat', 'detailed_portrait', 'kerry_james_marshall', 'medieval'], [
            0.08419699216085753, -0.07851817041961259, 0.056928033643688675, -0.028233440372052384, 0.09795210363137236,
            -0.02846836075845514
        ]

    def sources_one_hot(self):
        return ['human_prompts', 'r/Showerthoughts', 'r/EarthPorn', 'r/scifi', 'r/pics'], [
            0.08969932181089971, -0.010294043640662762, 0.09210472353549924, 0.0183836782636583, -0.03587177847683946,
            -0.05016474360675957
        ]

    def num_inference_steps_one_hot(self):
        return [
            100
        ], [
            0.07907408574691606, 0.024783072138878422
        ]

    def custom_aggregation(self, prefix, data):
        self.artist_styles_categories = ['van_gogh', 'jean-michel_basquiat', 'detailed_portrait',
                                         'kerry_james_marshall', 'medieval']
        self.sources_categories = ['human_prompts', 'r/Showerthoughts', 'r/EarthPorn', 'r/scifi', 'r/pics']
        result = {
            f'{prefix}_likes': np.sum((data['engagement_type'] == 'Like') & (data['engagement_value'] == 1)),
            f'{prefix}_dislikes': np.sum((data['engagement_type'] == 'Like') & (data['engagement_value'] == -1)),
            f'{prefix}_engagement_time_avg': data[data['engagement_type'] == 'MillisecondsEngagedWith'][
                'engagement_value'].mean(),
        }
        if prefix == 'user':
            for artist_style in self.artist_styles_categories:
                result.update(
                    {
                        f'{prefix}_{artist_style}_likes': np.sum(
                            (data['engagement_type'] == 'Like') & (data['engagement_value'] == 1) & (
                                    data['artist_style'] == artist_style)),
                        f'{prefix}_{artist_style}_dislikes': np.sum(
                            (data['engagement_type'] == 'Like') & (data['engagement_value'] == -1) & (
                                    data['artist_style'] == artist_style)),
                    }
                )
            for source in self.sources_categories:
                result.update(
                    {
                        f'{prefix}_{source}_likes': np.sum(
                            (data['engagement_type'] == 'Like') & (data['engagement_value'] == 1) & (
                                    data['source'] == source)),
                        f'{prefix}_{source}_dislikes': np.sum(
                            (data['engagement_type'] == 'Like') & (data['engagement_value'] == -1) & (
                                    data['source'] == source)),
                    }
                )
        return pd.Series(result)

    def feature_generation_user(self):
        aggregated_data = self.user_data.merge(self.generated_content_metadata_data, on=['content_id'], how='left').groupby('user_id').apply(lambda data: self.custom_aggregation('user', data))
        if 'user_id' in aggregated_data:
            aggregated_data = aggregated_data.drop('user_id', axis=1)
        return aggregated_data.reset_index()

    def coefficients(self):
        return {
            'content_likes': 0.0393806685305066,
            'content_dislikes': -0.1251121639248738,
            'content_engagement_time_avg': 1.7047921436141263e-11,

            'user_likes': 5.4764371246059006e-05,
            'user_dislikes': -0.00038144676253312266,
            'user_engagement_time_avg': 1.3979832806922074e-07,

            'user_van_gogh_likes': -0.0015617486488857127,
            'user_van_gogh_dislikes': 0.007679110213892672,
            'user_jean-michel_basquiat_likes': -0.0009773921103118932,
            'user_jean-michel_basquiat_dislikes': -0.0008799250096847402,
            'user_detailed_portrait_likes': -0.0017054374720878109,
            'user_detailed_portrait_dislikes': 0.0006448387863193699,
            'user_kerry_james_marshall_likes': 0.010037528570189526,
            'user_kerry_james_marshall_dislikes': 0.01263553743524961,
            'user_medieval_likes': -0.002036888897848471,
            'user_medieval_dislikes': 0.0067493455744942355,
            'user_human_prompts_likes': 0.00046769278872973313,
            'user_human_prompts_dislikes': -0.0007526283000997576,
            'user_r/Showerthoughts_likes': 0.0014862618080521658,
            'user_r/Showerthoughts_dislikes': -0.0035579735080894007,
            'user_r/EarthPorn_likes': -7.742487892404582e-05,
            'user_r/EarthPorn_dislikes': -0.0021741285998859673,
            'user_r/scifi_likes': 0.002639451581008141,
            'user_r/scifi_dislikes': 0.004024616643409577,
            'user_r/pics_likes': 0.0027597431193407715,
            'user_r/pics_dislikes': 0.006782971979724735,
        }

    def threshold(self):
        return 0.35

    def policy_filter_one(self, training_data, target_size=100):
        def filter_by_style(content):
            df = content.groupby('content_id').agg({'artist_style': lambda x: x.iloc[0], 'user_likes': 'sum'})

            proportions = df.artist_style.value_counts(normalize=True)
            proportions[proportions > 0.2] = 0.2
            P = proportions.sum()

            filtered_content_ids = []
            for cat, pro in proportions.items():
                filtered_content_ids.extend(
                    list(df[df.artist_style == cat].sort_values(by='user_likes', ascending=False)[
                         :int(pro * target_size / P)].index)
                )
            return filtered_content_ids

        return filter_by_style(training_data)

    def policy_filter_two(self, training_data, target_size=100):
        df = training_data.groupby('content_id').agg({'content_likes': 'sum', 'content_dislikes': 'sum'})

        def filter_by_likes(contents):
            contents['popular'] = contents.content_likes > 0
            value_counts = contents['popular'].value_counts(normalize=True)
            print(value_counts)
            p = value_counts.loc[True]
            q = value_counts.loc[False]

            a = 0.8 / p
            b = 0.2 / q

            filtered_content_ids = pd.concat([
                contents[contents.popular].sort_values(by='content_likes')[:int(a * target_size / (a + b))],
                contents[contents.popular].sort_values(by='content_likes')[:int(b * target_size / (a + b))]
            ]).index

            return list(filtered_content_ids)
        return filter_by_likes(df)


class BetaFilter(AbstractFilter):
    def _filter_ids(self, user_id, content_ids, seed, starting_point):
        dc = DataCollectorBeta()
        dc.gather_data(user_id, content_ids)
        dc.feature_eng()
        if starting_point.get("policy_filter_one", False):
            pf_one = dc.policy_filter_one(dc.results)  # policy one used here
        else:
            pf_one = set(content_ids)
        if starting_point.get("policy_filter_two", False):
            pf_two = dc.policy_filter_two(dc.results)  # policy two used here
        else:
            pf_two = set(content_ids)
        if starting_point.get("linear_model", False) and user_id not in [0, None]:
            pf_lr = set(dc.run_linear_model())
        else:
            pf_lr = set(content_ids)
        return pf_one & pf_two & pf_lr

    def _get_name(self):
        return "BetaFilter"
