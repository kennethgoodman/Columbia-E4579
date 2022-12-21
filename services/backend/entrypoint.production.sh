#!/bin/sh

echo "creating NNF"
python -m src.recommendation_system.recommendation_flow.utils.DeltaCfTask

echo "creating image scores"
python -m src.recommendation_system.recommendation_flow.utils.DeltaScoreTask

gunicorn -b 0.0.0.0:5000 wsgi:app --access-logfile - --workers 1 --threads 5 --timeout 120 --log-level=debug
