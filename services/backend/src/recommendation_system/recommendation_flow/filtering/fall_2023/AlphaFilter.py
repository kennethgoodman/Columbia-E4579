from src.recommendation_system.recommendation_flow.filtering.AbstractFilter import AbstractFilter
from src.recommendation_system.recommendation_flow.filtering.linear_model_helper import DataCollector
import random

class DataCollectorAlpha(DataCollector):
    def coefficients(self):
        return {
            'content_likes': 0.411,
            'content_dislikes': -1.25,
            'content_engagement_time_avg': -7.0458305856391235e-09,

            'user_likes': 0.00327,
            'user_dislikes': -0.0029,
            'user_engagement_time_avg': 1.6684465996563702e-06,
        }

    def artist_styles_one_hot(self):
        return [
            'van_gogh', 'jean-michel_basquiat', 'detailed_portrait',
            'kerry_james_marshall', 'medieval', 'studio', 'edward_hopper',
            'takashi_murakami', 'anime', 'leonardo_da_vinci',
            'laura_wheeler_waring', 'ma_jir_bo', 'jackson_pollock',
            'shepard_fairey', 'unreal_engine', 'face_and_lighting', 'keith_haring',
            'marta_minujín', 'franck_slama', 'oil_on_canvas', 'scifi', 'gta_v',
            'louise bourgeois', 'salvador_dali', 'ibrahim_el_salahi', 'juan_gris'
        ], [1.83523859e+00, 2.77885424e-01,
            1.70090565e+00, 9.49429353e-01, 2.15906078e+00, 2.44568696e+00,
            1.79434951e+00, 9.07959350e-01, 2.17337238e+00, 4.60592065e-01,
            4.09057593e-01, 8.42021422e-01, 6.68317933e-01, 1.76814775e+00,
            1.41639637e+00, 6.58195898e-01, 1.50787499e+00, 1.21004481e+00,
            3.05043625e-01, 1.94261633e+00, 1.96512518e+00, 9.10388431e-01,
            9.15045962e-01, 1.31553474e+00, 1.54801667e+00, 1.90712583e+00, 0]

    def sources_one_hot(self):
        return [
            'human_prompts', 'r/Showerthoughts', 'r/EarthPorn', 'r/scifi', 'r/pics',
            'r/Damnthatsinteresting', 'r/MadeMeSmile', 'r/educationalgifs',
            'r/SimplePrompts'
        ], [1.78062418e+00, -4.38372622e-01, 7.01822304e-01, -7.25890558e-02,
            -6.61610991e-01, -8.99656743e-01, -1.01408713e+00, -1.54566954e-01,
            2.05666798e-01, -0.5527702]

    def num_inference_steps_one_hot(self):
        return [
            100
        ], [
            0.535, -0.535
        ]

    def threshold(self):
        return 0.001

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

    def policy_filter_one(self, training_data, content_id):
        """
        Filtering on Likes: only keeping images with a 'decent' number of likes (or over a small negative number, say -1)
        """
        try:
            if training_data[training_data.content_id == content_id].content_engagement_time_avg.mean() > 1e3:
                # making sure people saw the img, say at least 1s
                like_value = training_data[training_data.content_id == content_id].content_likes.iloc[0]
                dislike_value = training_data[training_data.content_id == content_id].content_dislikes.iloc[0]
                diff = like_value - dislike_value
                # only keep first line since we only care about the nb of likes
                if diff > 0:
                    return True
                return False
            else:
                return True
        except Exception as e:
            print(f"got an an exception {e} in policy_filter_one for Alpha")
            # if no like value, that means
            return True

    def policy_filter_two(self, training_data, content_id):
        """ Checking if source is from the 'other' category;
        most of the movies pictures being in this category, we will return this type of image only x% of the time.
        The random package MUST be imported"""
        try:
            source = training_data[training_data.content_id == content_id].source.iloc[0]
            if source not in [
                    'human_prompts', 'r/Showerthoughts', 'r/EarthPorn', 'r/scifi', 'r/pics',
                    'r/Damnthatsinteresting', 'r/MadeMeSmile', 'r/educationalgifs',
                    'r/SimplePrompts'
                ]:
                random_number = random.random()

                # threshold for selection
                threshold = 0.90

                # Check if the random number is less than the probability threshold
                if random_number < threshold:
                    return True
            return False
        except Exception as e:
            print(f"got an an exception {e} in policy_filter_two for Alpha")
            return False


class AlphaFilter(AbstractFilter):
    def _filter_ids(self, user_id, content_ids, seed, starting_point):
        dc = DataCollectorAlpha()
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
        return "AlphaFilter"
