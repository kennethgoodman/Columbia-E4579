#!/bin/sh

echo "Waiting for mysql..."

while ! nc -z api-db 3306; do
  sleep 0.1
done

echo "mysql started"

echo "creating NNF"
python -m src.recommendation_system.recommendation_flow.utils.DeltaCfTask

echo "creating image scores"
python -m src.recommendation_system.recommendation_flow.utils.DeltaScoreTask

python manage.py run -h 0.0.0.0
