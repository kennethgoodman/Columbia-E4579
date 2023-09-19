
from .AbstractGenerator import AbstractGenerator
from .RandomGenerator import RandomGenerator

class YourChoiceGeneratorGenerator(AbstractGenerator):
    def _get_content_ids(self, _, limit, offset, _seed, starting_point):
        raise NotImplementedError("Need to implement this")
    
    def _get_name(self):
        return "YourChoiceGenerator"
