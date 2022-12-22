import heapq

from .AbstractRanker import AbstractRanker


class FoxtrotRanker(AbstractRanker):
    def rank_ids(self, limit, probabilities, seed, starting_point):
        k = limit
        top_t = heapq.nlargest(
            k, probabilities, key=lambda x: x["p_engage"]
        )  # find the largest k items with highest scores
        # shuffle top_t by switching selected two items
        # mapping with id
        top_t_ids = list(map(lambda x: x["content_id"], top_t))
        return top_t_ids
