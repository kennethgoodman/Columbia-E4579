import random

from .AbstractFilter import AbstractFilter


class RandomFilter(AbstractFilter):
    def filter_ids(self, content_ids, seed, starting_point):
        # choose 10% randomly
        if seed:
            random.seed(seed)
        return random.sample(content_ids, int(len(content_ids) * 0.1))
