from typing import List
import pandas as pd


class EchoRanker:

    def rank_ids(self, probabilities, limit=None, seed=None, starting_point=None) -> List[int]:

        probabilities = pd.DataFrame(probabilities, columns=['content_id', 'p_engage']).sort_values('p_engage', ascending=False)
        return probabilities['content_id'].tolist()[:limit]

