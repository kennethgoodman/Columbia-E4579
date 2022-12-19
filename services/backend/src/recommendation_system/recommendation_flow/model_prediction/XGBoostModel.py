
from typing import List
from .AbstractRanker import AbstractRanker
import heapq
from sqlalchemy.engine import create_engine
import pandas as pd
import numpy as np
import json

#from tqdm import tqdm,tqdm_notebook
from tqdm.notebook import tqdm
tqdm.pandas()

import re
import nltk
import string
from textblob import TextBlob
from collections import defaultdict
from collections import Counter
from nltk.corpus import wordnet
from nltk.corpus import stopwords

from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
from sklearn.preprocessing import OneHotEncoder

engine = create_engine('mysql://mysql:mysql@127.0.0.1:3307/api_dev')
content = pd.read_sql('select * from api_dev.content', engine)
engagement = pd.read_sql('select * from api_dev.engagement', engine)
gcm = pd.read_sql('select * from api_dev.generated_content_metadata', engine)
target = pd.read_csv("backend/processed_data/target.csv")

class XGBoostModel(AbstractModel):

    def decontraction(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"he's", "he is", phrase)
    phrase = re.sub(r"there's", "there is", phrase)
    phrase = re.sub(r"We're", "We are", phrase)
    phrase = re.sub(r"That's", "That is", phrase)
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"they're", "they are", phrase)
    phrase = re.sub(r"Can't", "Cannot", phrase)
    phrase = re.sub(r"wasn't", "was not", phrase)
    phrase = re.sub(r"don\x89Ûªt", "do not", phrase)
    phrase = re.sub(r"donãât", "do not", phrase)
    phrase = re.sub(r"aren't", "are not", phrase)
    phrase = re.sub(r"isn't", "is not", phrase)
    phrase = re.sub(r"What's", "What is", phrase)
    phrase = re.sub(r"haven't", "have not", phrase)
    phrase = re.sub(r"hasn't", "has not", phrase)
    phrase = re.sub(r"There's", "There is", phrase)
    phrase = re.sub(r"He's", "He is", phrase)
    phrase = re.sub(r"It's", "It is", phrase)
    phrase = re.sub(r"You're", "You are", phrase)
    phrase = re.sub(r"I'M", "I am", phrase)
    phrase = re.sub(r"shouldn't", "should not", phrase)
    phrase = re.sub(r"wouldn't", "would not", phrase)
    phrase = re.sub(r"i'm", "I am", phrase)
    phrase = re.sub(r"I\x89Ûªm", "I am", phrase)
    phrase = re.sub(r"I'm", "I am", phrase)
    phrase = re.sub(r"Isn't", "is not", phrase)
    phrase = re.sub(r"Here's", "Here is", phrase)
    phrase = re.sub(r"you've", "you have", phrase)
    phrase = re.sub(r"you\x89Ûªve", "you have", phrase)
    phrase = re.sub(r"we're", "we are", phrase)
    phrase = re.sub(r"what's", "what is", phrase)
    phrase = re.sub(r"couldn't", "could not", phrase)
    phrase = re.sub(r"we've", "we have", phrase)
    phrase = re.sub(r"it\x89Ûªs", "it is", phrase)
    phrase = re.sub(r"doesn\x89Ûªt", "does not", phrase)
    phrase = re.sub(r"It\x89Ûªs", "It is", phrase)
    phrase = re.sub(r"Here\x89Ûªs", "Here is", phrase)
    phrase = re.sub(r"who's", "who is", phrase)
    phrase = re.sub(r"I\x89Ûªve", "I have", phrase)
    phrase = re.sub(r"y'all", "you all", phrase)
    phrase = re.sub(r"can\x89Ûªt", "cannot", phrase)
    phrase = re.sub(r"would've", "would have", phrase)
    phrase = re.sub(r"it'll", "it will", phrase)
    phrase = re.sub(r"we'll", "we will", phrase)
    phrase = re.sub(r"wouldn\x89Ûªt", "would not", phrase)
    phrase = re.sub(r"We've", "We have", phrase)
    phrase = re.sub(r"he'll", "he will", phrase)
    phrase = re.sub(r"Y'all", "You all", phrase)
    phrase = re.sub(r"Weren't", "Were not", phrase)
    phrase = re.sub(r"Didn't", "Did not", phrase)
    phrase = re.sub(r"they'll", "they will", phrase)
    phrase = re.sub(r"they'd", "they would", phrase)
    phrase = re.sub(r"DON'T", "DO NOT", phrase)
    phrase = re.sub(r"That\x89Ûªs", "That is", phrase)
    phrase = re.sub(r"they've", "they have", phrase)
    phrase = re.sub(r"i'd", "I would", phrase)
    phrase = re.sub(r"should've", "should have", phrase)
    phrase = re.sub(r"You\x89Ûªre", "You are", phrase)
    phrase = re.sub(r"where's", "where is", phrase)
    phrase = re.sub(r"Don\x89Ûªt", "Do not", phrase)
    phrase = re.sub(r"we'd", "we would", phrase)
    phrase = re.sub(r"i'll", "I will", phrase)
    phrase = re.sub(r"weren't", "were not", phrase)
    phrase = re.sub(r"They're", "They are", phrase)
    phrase = re.sub(r"Can\x89Ûªt", "Cannot", phrase)
    phrase = re.sub(r"you\x89Ûªll", "you will", phrase)
    phrase = re.sub(r"I\x89Ûªd", "I would", phrase)
    phrase = re.sub(r"let's", "let us", phrase)
    phrase = re.sub(r"it's", "it is", phrase)
    phrase = re.sub(r"can't", "cannot", phrase)
    phrase = re.sub(r"don't", "do not", phrase)
    phrase = re.sub(r"you're", "you are", phrase)
    phrase = re.sub(r"i've", "I have", phrase)
    phrase = re.sub(r"that's", "that is", phrase)
    phrase = re.sub(r"i'll", "I will", phrase)
    phrase = re.sub(r"doesn't", "does not",phrase)
    phrase = re.sub(r"i'd", "I would", phrase)
    phrase = re.sub(r"didn't", "did not", phrase)
    phrase = re.sub(r"ain't", "am not", phrase)
    phrase = re.sub(r"you'll", "you will", phrase)
    phrase = re.sub(r"I've", "I have", phrase)
    phrase = re.sub(r"Don't", "do not", phrase)
    phrase = re.sub(r"I'll", "I will", phrase)
    phrase = re.sub(r"I'd", "I would", phrase)
    phrase = re.sub(r"Let's", "Let us", phrase)
    phrase = re.sub(r"you'd", "You would", phrase)
    phrase = re.sub(r"It's", "It is", phrase)
    phrase = re.sub(r"Ain't", "am not", phrase)
    phrase = re.sub(r"Haven't", "Have not", phrase)
    phrase = re.sub(r"Could've", "Could have", phrase)
    phrase = re.sub(r"youve", "you have", phrase)
    phrase = re.sub(r"donå«t", "do not", phrase)
    return phrase

    def remove_punctuations(text):
        for punctuation in list(string.punctuation):
            text = text.replace(punctuation, '')
        return text

    def clean_text(text):
        text = decontraction(text)
        text = text.lower()
        text = remove_punctuations(text)
        return text

    def text_features(df,col):
        df["num_words"] = df[col].progress_apply(lambda x: len(str(x).split()))
        df["num_chars"] = df[col].progress_apply(lambda x: len(str(x)))
        df["num_stopwords"] = df[col].progress_apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords.words('english')]))
        df["num_punctuations"] = df[col].progress_apply(lambda x: len([c for c in str(x) if c in list(string.punctuation)]))
        df["mean_word_len"] = df[col].progress_apply(lambda x: np.mean([len(w) for w in str(x).split()]))
        #Sentiment: lies between [-1,1], -1 defines a negative sentiment and 1 defines a positive sentiment.
        df["polarity"] = df[col].progress_apply(lambda x: TextBlob(x).sentiment[0])
        #The higher subjectivity means that the text contains personal opinion rather than factual information.
        df["subjectivity"] = df[col].progress_apply(lambda x: TextBlob(x).sentiment[1])
        return df

    def predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):

        # Text Feature

        gcm = gcm[gcm['content_id'].isin(content_ids)]

        gcm['original_prompt'] = gcm['original_prompt'].progress_apply(clean_text)

        gcm = text_features(gcm, "original_prompt")

        gcm["prompt_embedding"] = gcm["prompt_embedding"].progress_apply(lambda x:eval(x))

        # Generate Like Rate

        content_like = target.groupby("content_id").mean()["engagement_value"].rename("Like Rate").reset_index()

        gcm = pd.merge(gcm,content_like,on="content_id")


        return list(
            map(
                lambda content_id: {
                    "content_id": content_id,
                    "p_engage": random.random(),
                    "score": kwargs.get("scores", {})
                    .get(content_id, {})
                    .get("score", None),
                },
                content_ids,
            )
        )
