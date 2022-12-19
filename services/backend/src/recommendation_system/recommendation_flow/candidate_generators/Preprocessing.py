#!/usr/bin/env python
# coding: utf-8

# In[1]:
'''
generate_most_frequent_word(): Takes a dataframe containing prompts as input and output a column of most frequent words
in each prompt

get_sentiment_score(): Takes a dataframe containing prompts as input and output a column of compound sentiment score,
which is:
- Computed by summing the valence scores of each word in the lexicon
- Adjusted according to rules, and normalized to be between -1 (most extreme negative) and +1 (most extreme positive)
'''
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

class Preprocessing:
    def generate_most_frequent_word(df):
        def get_most_frequent_word(text):
            stop_words = stopwords.words('english')
            word_dict = dict()
            for i in set(text.lower().split(' ')):
                if i not in stop_words and i.isalpha():
                    word_dict[i] = text.count(i)
            if not word_dict:
                return None
            return max(word_dict, key=lambda key: word_dict[key])
        df['frequent_word'] = df['prompt'].apply(lambda x: get_most_frequent_word(x))
        return df
    
    def get_sentiment_score(df):
        sentimentAnalyser = SentimentIntensityAnalyzer()
        df['sentiment'] = df['prompt'].apply(lambda x: sentimentAnalyser.polarity_scores(x)['compound'])
        return df

