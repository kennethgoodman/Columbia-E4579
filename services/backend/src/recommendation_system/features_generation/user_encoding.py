import pandas as pd

eng_df = pd.read_csv("services/backend/seed_data/data/nov_19_dump/engagement.csv")

user_id = pd.DataFrame(eng_df['user_id'].unique(), columns=['user_id'])
user_id = user_id.sort_values('user_id')
user_id_dummy = pd.get_dummies(user_id['user_id'], prefix='user')
df = pd.concat([user_id, user_id_dummy], axis=1)

df.to_csv('services/backend/processed_data/user_embedding.csv', index=False)