import random

from .AbstractFilter import AbstractFilter


class RandomFilter(AbstractFilter):
    def _filter_ids(self, user_id, content_ids, seed, starting_point, amount=0.1, dc=None):
        # choose 10% randomly
        if seed:
            random.seed(seed)
        return random.sample(content_ids, int(len(content_ids) * amount))

    def _get_name(self):
        return "RandomFilter"
