import os
import pickle
from functools import lru_cache

import mrpt
import numpy as np
from src.api.content.models import Content, GeneratedContentMetadata

INDEXES = {}  # (target recall => index)
INDEX_TO_CONTENT_ID = {}
CONTENT_ID_TO_INDEX = {}


@lru_cache(1)
def read_data():
    global INDEX_TO_CONTENT_ID
    if os.path.isfile("/usr/src/app/id_to_embedding.pkl"):
        print("reading data from id_to_embedding.pkl")
        with open("/usr/src/app/id_to_embedding.pkl", "rb") as f:
            data = pickle.load(f)
    else:
        data = GeneratedContentMetadata.query.with_entities(
            GeneratedContentMetadata.content_id,
            GeneratedContentMetadata.prompt_embedding,
        ).all()
    np_data = []
    i = 0
    if len(data) == 0:
        raise ValueError(
            """
            You probably don't have the prompt_to_embedding file:
            https://github.com/kennethgoodman/Columbia-E4579/blob/main/services/backend/seed_data/data/prompt_to_embedding.64.100.1000.pkl
            You have 2 options:
            1) manually download the file from the link above and put it in /services/backend/seed_data/data/ folder
            2) install git lfs and use that to pull the file, running "git lfs pull" (preferred but more effort)
        """
        )
    for (content_id, embedding) in data:
        if embedding is None:
            continue
        INDEX_TO_CONTENT_ID[i] = content_id
        CONTENT_ID_TO_INDEX[content_id] = i
        i += 1
        np_data.append(np.array(embedding, dtype=np.float32))
    return np.array(np_data)


def instantiate(target_recall, k=25):  # instantiate k=25, but can ask for more later
    global INDEXES
    data = read_data()
    index = mrpt.MRPTIndex(data)
    index.build_autotune_sample(target_recall, k)
    INDEXES[target_recall] = index


def get_embedding(content_id):
    # Explore keeping data in memory for all embeddings if this is too slow
    return np.array(
        GeneratedContentMetadata.query.with_entities(
            GeneratedContentMetadata.prompt_embedding
        )
        .filter_by(content_id=content_id)
        .first(),
        dtype=np.float32,
    )


def ann(content_id, target_recall, k=25, return_distances=False):
    global INDEXES, INDEX_TO_CONTENT_ID
    model = INDEXES[target_recall]
    idx = CONTENT_ID_TO_INDEX.get(content_id, None)
    if idx is None:
        return None, None
    q = read_data()[idx]
    rtn = model.ann(q, k=k, return_distances=return_distances)
    scores = None
    if return_distances:
        data_indexes, scores = rtn
    else:
        data_indexes = rtn
    content_ids = list(
        filter(None, map(lambda index: INDEX_TO_CONTENT_ID.get(index), data_indexes))
    )
    return content_ids, scores


def ann_with_offset(content_id, target_recall, limit, offset, return_distances=False):
    content_ids, scores = ann(
        content_id, target_recall, k=limit + offset, return_distances=return_distances
    )
    if content_ids is None:
        return [], []
    if offset == 0 and content_ids[0] != content_id:
        content_ids = [content_id] + content_ids
    return content_ids[offset:], (scores[offset:] if scores is not None else None)
