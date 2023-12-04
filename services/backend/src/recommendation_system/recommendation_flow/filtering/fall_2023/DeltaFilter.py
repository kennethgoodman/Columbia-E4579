from src.recommendation_system.recommendation_flow.filtering.AbstractFilter import AbstractFilter
from src.recommendation_system.recommendation_flow.filtering.linear_model_helper import AbstractFeatureEng


class FeatureEngDelta(AbstractFeatureEng):
    def coefficients(self):
        return {
            'content_likes': 0.008504499974045877,
            'content_dislikes': -0.018620360912729533,
            'content_engagement_time_avg': 1.939212871998199e-10,
            'user_likes': 4.6184837227071535e-05,
            'user_dislikes': -5.319159929086619e-05,
            'user_engagement_time_avg': 9.385624489095923e-08,
        }

    def artist_styles_one_hot(self):
        artstyles_no_movie = [
            'shepard_fairey',
            'studio',
            'medieval',
            'unreal_engine',
            'edward_hopper',
            'anime',
            'kerry_james_marshall',
            'oil_on_canvas',
            'detailed_portrait',
            'gta_v',
            'scifi',
            'van_gogh',
            'salvador_dali',
            'jean-michel_basquiat',
            'face_and_lighting']
        style_coeffs = [
            0.004409142545343398,
            0.011894449166011137,
            0.014907388137700846,
            0.0008096059519425987,
            0.006714741985170694,
            0.009934734759402422,
            -0.014606625603665805,
            0.00794564982604023,
            -0.0007318421884443975,
            -0.005439435739255107,
            0.010627647921063176,
            0.008557881286745164,
            -0.0142722167119279,
            -0.013029608369880893,
            -0.012093821762478144,
            -0.01562769120372921]

        return artstyles_no_movie, style_coeffs

    def sources_one_hot(self):
        sources = ['human_prompts',
                   'r/Showerthoughts',
                   'r/EarthPorn']
        source_coeffs = [
            0.015505005789178464,
            -0.010298992024823862,
            0.0,
            -0.005206013764405199]

        return sources, source_coeffs

    def num_inference_steps_one_hot(self):
        steps = [20, 50, 100]
        step_coeffs = [
            -7.50395834568939e-05,
            0.0,
            0.0048317668567876846,
            -0.004756727273321751]
        return steps, step_coeffs

    def threshold(self):
        return -0.15

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
        filtered_data = training_data[training_data['content_id'].isin(content_ids)].drop_duplicates(subset=['content_id'])
        seen_ids = filtered_data['content_id'].to_list()
        content_id_set = set(content_ids)
        #list of ids not seen in training data
        unseen_ids = [id for id in content_id_set if id not in seen_ids]

        #calculated desired number of content to remove
        desired_remove_count = int(len(content_id_set) * 0.82)

        #calculate top 35 quantile avg eng time across contents
        top_eng = training_data['content_engagement_time_avg'].quantile(.65)

        data_to_remove = filtered_data.loc[filtered_data['content_engagement_time_avg'] < top_eng]
        ids_to_remove = data_to_remove['content_id'].to_list()
        remove_desired_diff = abs(len(ids_to_remove) - desired_remove_count)

        if len(ids_to_remove) < desired_remove_count:
          #if removed less than desired, remove extra from unseen ids and seen ids based on dislikes
          #remove at least one third of remaining seen ids, the ones with highest dislikes
          seen_remained_count = len(seen_ids) - len(ids_to_remove)
          seen_remove_count = max(int(seen_remained_count * 0.3), remove_desired_diff - len(unseen_ids))
          seen_to_remove = filtered_data.loc[filtered_data['content_engagement_time_avg'] >= top_eng].nlargest(seen_remove_count,'content_dislikes', keep='first')['content_id'].to_list()
          ids_to_remove += seen_to_remove

          unseen_remove_count = desired_remove_count - len(ids_to_remove)
          ids_to_remove += unseen_ids[:unseen_remove_count]

        else:
          #if removed more than desired, remove less from filtered data based on content_likes
          top_k = data_to_remove.nlargest(remove_desired_diff, 'content_likes', keep='first')['content_id'].to_list()
          ids_to_remove = [id for id in ids_to_remove if id not in top_k]

        filtered_ids = [index for index in content_id_set if index not in ids_to_remove]

        return filtered_ids



    def policy_filter_two(self, training_data, content_ids):
        filtered_data = training_data[training_data['content_id'].isin(content_ids)].drop_duplicates(subset=['content_id'])
        seen_ids = filtered_data['content_id'].to_list()
        content_id_set = set(content_ids)
        #list of ids not seen in training data
        unseen_ids = [id for id in content_id_set if id not in seen_ids]

        #calculated desired number of content to remove
        half_desired_remove_count = int(len(content_id_set) * 0.03)

        #remove content with highest dislikes (half of desired remove count)
        ids_to_remove = set(filtered_data.nlargest(half_desired_remove_count,'content_dislikes', keep='first')['content_id'])

        #remove content with lowest likes from remaining content having specific artstyles (half of desired remove count)
        exclude_artstyles = ['movie: Batman', 'scifi', 'laura_wheeler_waring', 'marta_minujín','kerry_james_marshall', 'jean-michel_basquiat', 'movie: Indiana-Jones-IV', 'unreal_engine', 'anime']
        ids_to_remove.update(filtered_data[filtered_data['artist_style'].isin(exclude_artstyles) & ~filtered_data['content_id'].isin(ids_to_remove)].nsmallest(half_desired_remove_count, 'content_likes', keep='first')['content_id'])

        filtered_ids = [index for index in content_id_set if index not in ids_to_remove]

        return filtered_ids


class DeltaFilter(AbstractFilter):
    def _filter_ids(self, user_id, content_ids, seed, starting_point, amount=None, dc=None):
        delta_feature_eng = FeatureEngDelta(dc)
        delta_feature_eng.feature_eng()
        if starting_point.get("policy_filter_one", False):
            pf_one = delta_feature_eng.policy_filter_one(delta_feature_eng.results, set(content_ids))
        else:
            pf_one = set(content_ids)
        if starting_point.get("policy_filter_two", False):
            pf_two = delta_feature_eng.policy_filter_two(delta_feature_eng.results, pf_one)
        else:
            pf_two = pf_one
        if starting_point.get("linear_model", False) and user_id not in [0, None]:
            pf_lr = set(delta_feature_eng.run_linear_model(pf_two))
        else:
            pf_lr = pf_two
        return pf_lr

    def _get_name(self):
        return "DeltaFilter"
