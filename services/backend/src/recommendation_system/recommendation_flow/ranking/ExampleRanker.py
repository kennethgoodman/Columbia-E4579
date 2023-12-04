import heapq
import random

from .AbstractRanker import AbstractRanker


class ExampleRanker(AbstractRanker):
    def _rank_ids(self, user_id, content_ids, limit, probabilities, seed, starting_point, X=None):
        k = min(limit, len(content_ids))
        to_rank = list(zip(content_ids, probabilities[2]))
        top_k = heapq.nlargest(k, to_rank, lambda x: x[1])
        top_k_ids = list(map(lambda x: x[0], top_k))
        return top_k_ids

    def _get_name(self):
        return "RandomRanker"
