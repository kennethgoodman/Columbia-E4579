
from .AbstractGenerator import AbstractGenerator
from .RandomGenerator import RandomGenerator

class CollaberativeFilteredSimilarUsersGeneratorGenerator(AbstractGenerator):
    def _get_content_ids(self, _, limit, offset, _seed, starting_point):
        raise NotImplementedError("Need to implement this")
    
    def _get_name(self):
        "CollaberativeFilteredSimilarUsersGenerator"
