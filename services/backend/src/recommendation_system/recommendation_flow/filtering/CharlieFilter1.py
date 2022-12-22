import random
from src.api.engagement.crud import *

from .AbstractFilter import AbstractFilter


class DislikeRatioFilter(AbstractFilter):
    def filter_ids(self, content_ids, seed, starting_point):
        "Filter contents with dislike percentage more than 0.5"
        result = []
        for c_id in content_ids:
            dislike_count = get_dislike_count_by_content_id(c_id)
            like_count = get_like_count_by_content_id(c_id)
            if dislike_count is None:
                dislike_count = 0
            if like_count is None:
                like_count = 0
            if dislike_count + like_count > 0:
                dislike_ratio = dislike_count/(dislike_count+like_count) 
            else:
                dislike_ratio = 0 
            if dislike_ratio <= 0.50:
                result.append(c_id)
        return result
