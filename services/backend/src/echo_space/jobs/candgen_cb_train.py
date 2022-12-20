from sklearn import linear_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pickle
from joblib import dump

model = linear_model.LogisticRegression(max_iter=200)

train_df = pd.read_csv('services/backend/processed_data/train.csv')
test_df = pd.read_csv('services/backend/processed_data/test.csv')
user_embedding = pd.read_csv('services/backend/processed_data/user_embedding.csv', index_col=0)
item_embedding = pd.read_csv('services/backend/processed_data/embedding_pca_64.csv', index_col=0)

user_features = user_embedding.columns.tolist()
item_features = item_embedding.columns.tolist()

train_df = pd.merge(train_df, user_embedding, how="inner", on="user_id")
train_df = pd.merge(train_df, item_embedding, how="inner", on="content_id")
train_df = train_df[user_features + item_features + ['engagement_value']]

test_df = pd.merge(test_df, user_embedding, how="inner", on="user_id")
test_df = pd.merge(test_df, item_embedding, how="inner", on="content_id")
test_df = test_df[user_features + item_features + ['engagement_value']]

sc = StandardScaler()
train_df[item_features] = sc.fit_transform(train_df[item_features])
test_df[item_features] = sc.transform(test_df[item_features])

model.fit(train_df[user_features + item_features], train_df['engagement_value'])

pred_train = model.predict(train_df[user_features + item_features])
pred_test = model.predict(test_df[user_features + item_features])

cfr_train = classification_report(train_df['engagement_value'], pred_train)
cfr_test = classification_report(test_df['engagement_value'], pred_test)

print(cfr_test)

with open('services/backend/src/recommendation_system/ml_models/cb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

dump(sc, 'services/backend/src/recommendation_system/features_generation/processors/scaler_item_cg.bin',
     compress=True)
