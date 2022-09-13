#!/bin/sh

echo "Waiting for mysql..."

while ! nc -z api-db 3306; do
  sleep 0.1
done

echo "mysql started"

echo "recreating db"
python manage.py recreate_db
echo "db created"
python manage.py seed_db
echo "db seeded"

python manage.py run -h 0.0.0.0
