import pandas as pd

a = pd.read_csv('services/backend/seed_data/data/nov_19_dump/engagement.csv')
# print(a.content_id.nunique())
print(len(a))