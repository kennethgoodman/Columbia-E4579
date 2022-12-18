import os
import pickle
from functools import lru_cache

import mrpt
import numpy as np
from src.api.content.models import Content, GeneratedContentMetadata

OFFENSIVE_BLACKLIST = []


@lru_cache(1)
def read_data():
    # OFFENSIVE_BLACKLIST: a global variable that includes content_ids
    # that possibly include offensive 
    global OFFENSIVE_BLACKLIST 
    if os.path.isfile("df_prompt_labeled.pkl"):
        print("reading NLP label data")
        with open("df_prompt_labeled.pkl", "rb") as f:
            data = pickle.load(f)
            OFFENSIVE_BLACKLIST = list(data.loc[data.offensive == True].index)
    else:
        print("The NLP Label data is missing")

    return 0



