import heapq
import random

from .AbstractRanker import AbstractRanker


class AlphaRanker(AbstractRanker):
    def rank_ids(self, limit, probabilities, seed, starting_point):
        k = int(limit * 0.9)
        other = limit - k
        top_limit = heapq.nlargest(limit, probabilities, key=lambda x: x["p_engage"])
        top_k = top_limit[:k]
        try: # in dev no style
            style = set([i['style'] for i in top_k if i['style'] == i['style']])
            others = []
            for i in probabilities:
                if i['style'] not in style:
                    others.append(i)
            top_other = heapq.nlargest(other, others, key=lambda x: x["explore"])
            top_all = top_k + top_other
        except KeyError:
            top_all = top_k
        top_all_ids = list(map(lambda x: x["content_id"], top_all))
        
        if seed:
            random.seed(seed)
        return random.sample(top_all_ids, len(top_all_ids))

