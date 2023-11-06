from src.recommendation_system.recommendation_flow.filtering.AbstractFilter import AbstractFilter
from src.recommendation_system.recommendation_flow.filtering.linear_model_helper import DataCollector


class DataCollectorDelta(DataCollector):
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
        filtered_data = training_data[training_data['content_id'].isin(content_ids)].drop_duplicates(
            subset=['content_id'])

        ## calculate the number of contents for each artist_style
        src_counts = filtered_data["source"].value_counts().to_dict()

        keep_sources = ['human_prompts', 'Dances-with-Wolves', 'r/Damnthatsinteresting', 'r/MadeMeSmile',
                        'Johann Wolfgang von Goethe', 'r/AccidentalArt', 'Laozi']

        # Calculate the total number of contents
        total_contents = len(content_ids)

        # Calculate the maximum allowable count for sources
        # max_allowed_count = int(total_contents * 0.00001)
        max_allowed_count = 0

        ids_to_remove = []
        # Iterate through artists and check their counts
        for src in src_counts:
            if src not in keep_sources:
                src_count = src_counts[src]
                if (src_count > max_allowed_count):
                    remove_count = src_count - max_allowed_count
                    ids_to_remove.extend(
                        filtered_data[filtered_data['source'] == src].sample(remove_count)['content_id'])
                    # print(len(indices_to_remove))

        condition_1 = filtered_data['source'] == 'human_prompts'
        condition_2 = ~filtered_data['guidance_scale'].isin([7, 10, 15, 17])
        ids_to_remove.extend(filtered_data[condition_1 & condition_2]['content_id'])

        # change the ratio based on how much linear is expected to filter
        # if we are dropping too much, we should not do so
        stop_index = len(ids_to_remove) + 1
        # if len(indices_to_remove) / len(content_ids) > 3/13:
        #   stop_index = int(len(content_ids) * 3/13)

        remove_set = set(ids_to_remove[:stop_index])
        filtered_indices = [id for id in content_ids if id not in remove_set]

        return filtered_indices

    def policy_filter_two(self, training_data, content_ids):
        filtered_data = training_data[training_data['content_id'].isin(content_ids)].drop_duplicates(
            subset=['content_id'])

        ## calculate the number of contents for each artist_style
        artstyle_counts = filtered_data["artist_style"].value_counts().to_dict()

        artstyles = ['movie: Batman', 'scifi', 'laura_wheeler_waring', 'marta_minujín', 'kerry_james_marshall',
                     'jean-michel_basquiat', 'movie: Indiana-Jones-IV', 'unreal_engine', 'anime']

        # Calculate the total number of contents
        total_contents = len(content_ids)

        # Calculate the maximum allowable count for the artists
        max_allowed_count = int(total_contents * 0.00001)

        indices_to_remove = []
        # Iterate through artists and check their counts
        for style in artstyles:
            if style in artstyle_counts:
                style_count = artstyle_counts[style]
                if (style_count > max_allowed_count):
                    remove_count = style_count - max_allowed_count
                    indices_to_remove.extend(
                        filtered_data[filtered_data['artist_style'] == style].sample(remove_count).index)

        # change the ratio based on how much linear is expected to filter
        # if we are dropping too much, we should not do so
        stop_index = len(indices_to_remove) + 1
        # if len(indices_to_remove) / len(content_ids) > 3/13:
        #   stop_index = int(len(content_ids) * 3/13)

        remove_set = set(indices_to_remove[:stop_index])
        filtered_indices = [index for index in content_ids if index not in remove_set]

        return filtered_indices


class DeltaFilter(AbstractFilter):
    def _filter_ids(self, user_id, content_ids, seed, starting_point):
        dc = DataCollectorDelta()
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
        return "DeltaFilter"
