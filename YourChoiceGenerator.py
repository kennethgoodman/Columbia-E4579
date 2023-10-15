
from src.recommendation_system.recommendation_flow.candidate_generators.AbstractGenerator import AbstractGenerator
import numpy as np

#random generator

class YourChoiceGenerator(AbstractGenerator):
    def _get_content_ids(self, _, limit, offset, _seed, starting_point):
        content_id = np.random.randint(low=30001,high=100000,size=(1000,))
        content_id = content_id.tolist()
        return content_id, []
    
    def _get_name(self):
        return "YourChoiceGenerator"