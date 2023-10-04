
from src.recommendation_system.recommendation_flow.candidate_generators.AbstractGenerator import AbstractGenerator

class YourChoiceGenerator(AbstractGenerator):
    def _get_content_ids(self, _, limit, offset, _seed, starting_point):
        return [], []
    
    def _get_name(self):
        return "YourChoiceGenerator"
