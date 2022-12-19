
from typing import List
from .AbstractRanker import AbstractRanker
import heapq
from sqlalchemy.engine import create_engine
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

engine = create_engine('mysql://mysql:mysql@127.0.0.1:3307/api_dev')
engagement = pd.read_sql('select * from api_dev.engagement', engine)
metadata = pd.read_sql('select * from api_dev.generated_content_metadata', engine)
