import random

from .AbstractFilter import AbstractFilter


class RandomFilter(AbstractFilter):
    def filter_ids(self, content_ids):
        # choose 10% randomly
        return random.sample(content_ids, int(len(content_ids) * 0.1))
