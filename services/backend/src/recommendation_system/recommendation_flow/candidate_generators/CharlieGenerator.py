from sqlalchemy import text
from sqlalchemy.sql.expression import func
from src import db
import numpy as np
from src.api.content.models import Content
from src.data_structures.approximate_nearest_neighbor import ann_with_offset
from src.data_structures.approximate_nearest_neighbor import read_data
from src.data_structures.approximate_nearest_neighbor import get_embedding
from .AbstractGenerator import AbstractGenerator
from src.api.content.models import Content, GeneratedContentMetadata
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
from transformers import pipeline

sentiment_score = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
#sentiment_score("Covid cases are increasing fast!")
#data=read_data()
#data['sen_score']=df['prompt_embedding'].apply(sentiment_score)

def get_prompt(content_id):
    # Explore keeping data in memory for all embeddings if this is too slow
    return np.array(
        GeneratedContentMetadata.query.with_entities(
            GeneratedContentMetadata.prompt
        )
        .filter_by(content_id=content_id)
        .first(),
        dtype=np.float32,
    )

class RandomGenerator(AbstractGenerator):
    def get_content_ids(self, user_id, limit, offset, seed, starting_point):
        if starting_point is None:
            Content_ids = (
                Content.query.with_entities(Content.id)
                .order_by(func.random(seed))
                .limit(limit)
                .offset(offset)
                .all()
            )
            results=[]
            for cid in Content_ids:
                if sentiment_score(get_prompt(cid))<50:
                    results.append(cid)
            return list(map(lambda x: x[0], results)), None
        elif starting_point.get("content_id", False):
            content_ids, scores = ann_with_offset(
                starting_point["content_id"], 0.9, limit, offset, return_distances=True
            )
            return content_ids, scores
        raise NotImplementedError("Need to provide a key we know about")
