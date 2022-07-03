from . import AbstractRanker
import heapq


class RandomRanker(AbstractRanker):
    def rank_ids(self, probabilities):
        k = 10
        top_k = heapq.nlargest(k, probabilities, key=lambda x: x['p_engage'])
        return sorted(top_k, key=lambda x: x['p_engage'])
