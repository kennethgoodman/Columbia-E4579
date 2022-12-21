'''
BetaRanker is basically derived from the ExampleRanker presneted in class.

The ranker includes two shuffle methods:
1. Full random Shuffle will shuffle all top ids
2. One swap will swap the second photo and the last photo and keep the order of other photos

The two methods can be applied at the same time, or be easily disabled by commenting.
The file should be included in services/backend/src/recommendation_system/recommendation_flow/ranking

Thanks,
Qinghao Lin
'''

import heapq
import random


from .AbstractRanker import AbstractRanker

class BetaRanker(AbstractRanker):
    def rank_ids(self, limit, probabilities, seed, starting_point):
        k = limit
        # top_k = heapq.nlargest(k, probabilities, key=lambda x: x["p_engage"] + random.randint(25,75)/100)
        top_k = heapq.nlargest(k, probabilities, key=lambda x: x["p_engage"])


        top_k_ids = list(map(lambda x: x["content_id"], top_k))

        
        
        return top_k_ids
    
    
