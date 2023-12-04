from src.recommendation_system.recommendation_flow.filtering.AbstractFilter import AbstractFilter
from src.recommendation_system.recommendation_flow.filtering.linear_model_helper import AbstractFeatureEng


class FeatureEngFoxtrot(AbstractFeatureEng):
    def coefficients(self):
        return {
            'content_likes': 0.004913674532861648,
            'content_dislikes': -0.0025596352777526488,
            'content_engagement_time_avg': -1.1359359152350807e-10,
            'user_likes': 0.003527374532861648,
            'user_dislikes': -7.111467664335854e-03,
            'user_engagement_time_avg': -4.198888030205803e-09,
        }

    def artist_styles_one_hot(self):
        return ['medieval', 'anime', 'studio', 'oil_on_canvas', 'unreal_engine', 'edward_hopper', 'shepard_fairey', ], [
            0.00297867123567935, -0.00013583832416803058, 0.0011875140593540518, -0.0008508512202171477,
            -0.0011910319715602647, -0.0009473978012908358, 0.0015827575817499537, -0.002623823559547084]

    def sources_one_hot(self):
        return ['human_prompts', 'r/EarthPorn', 'r/Showerthoughts', 'r/scifi', 'r/pics', 'r/educationalgifs',
                'r/Damnthatsinteresting'], [
            0.003435938296346102, 0.002055277050584419, -0.0016261067852378207, -1.3188188316680503e-05,
            -0.0014536933603578126,
            -0.000850669870952967, -0.0013183930372200371, -0.00022916410484521567]

    def num_inference_steps_one_hot(self):
        return [20, 50, 100], [0.0003241249776025939, 0.00013478447734715756, 0.0015154732174153412,
                               -0.0019743826723650913]

    def threshold(self):
        return 0.005

    def policy_filter_one(self):
        temp = self.engagement_data.loc[self.engagement_data["engagement_type"] == "Like"].groupby(
            "content_id"
        )['engagement_value'].sum().rename('score', inplace=True).to_frame()
        temp = temp.loc[temp['score'] > 0]
        self.results = self.results[self.results['content_id'].isin(temp.index)].reset_index(drop=True)

    def policy_filter_two(self):
        temp2 = self.engagement_data.loc[self.engagement_data[
                                             'engagement_type'] == 'MillisecondsEngagedWith'].groupby(
            'content_id')['engagement_value'].count().rename(
            'count', inplace=True).to_frame().sort_values(by='count').iloc[50:]
        self.results = self.results[self.results['content_id'].isin(temp2.index)].reset_index(drop=True)


class FoxtrotFilter(AbstractFilter):
    def _filter_ids(self, dc, user_id, content_ids, seed, starting_point):
        foxtrot_feature_eng = FeatureEngFoxtrot(dc)
        foxtrot_feature_eng.feature_eng()
        if starting_point.get("policy_filter_one", False):
            foxtrot_feature_eng.policy_filter_one()  # policy one used here
        if starting_point.get("policy_filter_two", False):
            foxtrot_feature_eng.policy_filter_two()  # policy two used here
        if starting_point.get("linear_model", False) and user_id not in [0, None]:
            return set(foxtrot_feature_eng.run_linear_model())
        return set(foxtrot_feature_eng.results['content_id'].values)

    def _get_name(self):
        return "FoxtrotFilter"
