import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

content_df = pd.read_csv("services/backend/seed_data/data/nov_19_dump/content.csv")
eng_df = pd.read_csv("services/backend/seed_data/data/nov_19_dump/engagement.csv")
content_meta_df = pd.read_csv("services/backend/seed_data/data/nov_19_dump/generated_content_metadata.csv")

target_df = pd.read_csv('services/backend/processed_data/target.csv')
train_df = pd.read_csv('services/backend/processed_data/train.csv')
test_df = pd.read_csv('services/backend/processed_data/test.csv')

items_train = train_df['content_id'].unique().tolist()

content_meta_df["prompt_embedding"] = content_meta_df["prompt_embedding"].apply(lambda x:eval(x))
dft = pd.DataFrame(content_meta_df["prompt_embedding"].tolist())
dft = dft.add_prefix('prompt_embedding_')
features = list(dft.columns)
feature_df = pd.concat([content_meta_df['content_id'], dft], axis=1)

feature_df_train = feature_df[feature_df['content_id'].isin(items_train)]

sc = StandardScaler()
pca = PCA(n_components=64)

feature_df_train[features] = sc.fit_transform(feature_df_train[features])
pca.fit(feature_df_train[features])

feature_df_pca = pca.transform(feature_df[features])

embedding_pca_64 = pd.DataFrame(feature_df_pca)
embedding_pca_64 = embedding_pca_64.add_prefix('emv_pca_')
embedding_pca_64 = pd.concat([content_meta_df['content_id'], embedding_pca_64], axis=1)

embedding_pca_64.to_csv('services/backend/processed_data/embedding_pca_64.csv', index=False)
