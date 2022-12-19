from sklearn import linear_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pickle
from itertools import product
from joblib import load
from tqdm import tqdm
import json

with open('services/backend/src/recommendation_system/ml_models/cb_model.pkl', 'rb') as f:
    model = pickle.load(f)
sc = load('services/backend/src/recommendation_system/features_generation/processors/scaler_item_cg.bin')

eng_df = pd.read_csv("services/backend/seed_data/data/nov_19_dump/engagement.csv")
user_embedding = pd.read_csv('services/backend/processed_data/user_embedding.csv', index_col=0)
item_embedding = pd.read_csv('services/backend/processed_data/embedding_pca_64.csv', index_col=0)

user_ids = user_embedding.index.tolist()
item_ids = item_embedding.index.tolist()

item_seen = eng_df['content_id'].unique()
user_item_df = pd.DataFrame(list(product(user_ids, item_ids)), columns=['user_id', 'content_id'])
user_item_df = user_item_df[~user_item_df['content_id'].isin(item_seen)]

user_features = user_embedding.columns.tolist()
item_features = item_embedding.columns.tolist()

user_item_feat_df = pd.merge(user_item_df, user_embedding, how="inner", on="user_id")
user_item_feat_df = pd.merge(user_item_feat_df, item_embedding, how="inner", on="content_id")
user_item_feat_df[item_features] = sc.transform(user_item_feat_df[item_features])

recs = {}

for user_id in tqdm(user_ids):
    dft = user_item_feat_df[user_item_feat_df['user_id'] == user_id].copy()
    pred = model.predict_proba(dft[user_features + item_features])[:, 1]
    dft['score'] = pred
    dft = dft.sort_values('score', ascending=False)
    top_k = dft['content_id'].tolist()[:1500]

    recs[user_id] = top_k

    with open('services/backend/output/cg_cb_recs.json', 'w') as fp:
        json.dump(recs, fp)

