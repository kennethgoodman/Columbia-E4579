#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import heapq
import random

from .AbstractRanker import AbstractRanker


class RuleBasedRanker(AbstractRanker):
    def rank_ids(self, limit, predictions, seed, starting_point):
        k = limit #total number to show
        
        #80% from score=2, 10% from score=1, 10% from score =0
        score2k=int(0.8*k)
        score1k=int(0.1*k)
        score0k=int(0.1*k)
        
        prediction2=[]
        prediction1=[]
        prediction0=[]
        
        #split to 3 group by score
        for d in predictions:
            if d.get("p_engage")==2:
                prediction2.append(d)
            elif d.get("p_engage")==1:
                prediction1.append(d)
            elif d.get("p_engage")==0:
                prediction0.append(d)
        
        #select best of each 3 group
        zero_if_none = lambda x : float(x["score"]) if x["score"] is not None else 0.0
        top_score2k = heapq.nlargest(score2k, prediction2, key=zero_if_none) # sort by score
        top_score1k = heapq.nlargest(score1k, prediction1, key=zero_if_none)
        top_score0k = heapq.nlargest(score0k, prediction0, key=zero_if_none)
        top_score2k_ids = list(map(lambda x: x["content_id"], top_score2k))
        top_score1k_ids = list(map(lambda x: x["content_id"], top_score1k))
        top_score0k_ids = list(map(lambda x: x["content_id"], top_score0k))

        #add selected img together 
        top_k_ids=top_score2k_ids+top_score1k_ids+top_score0k_ids
        
        return random.sample(top_k_ids, len(top_k_ids))
