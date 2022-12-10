from services.backend.src.recommendation_system.ml_models.als import ALSEstimator
import pandas as pd



rating = pd.read_csv('../../../services/backend/processed_data/target.csv')

als = ALSEstimator()
als.fit(rating)

print(als.recommend(n_recommendations=10))
