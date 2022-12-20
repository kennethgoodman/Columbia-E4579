from services.backend.src.echo_space.features_generation.als import ALSEstimator
import pandas as pd
import json

rating = pd.read_csv('services/backend/processed_data/target.csv')

als = ALSEstimator(iterations=30)
als.fit(rating)
recs = als.recommend(n_recommendations=3500)

with open('services/backend/output/cg_cf_recs.json', 'w') as fp:
    json.dump(recs, fp)
