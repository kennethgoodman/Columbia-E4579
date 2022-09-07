# Columbia-E4579
Fall 2022 Class At Columbia. Modern Recommendation Systems

Install with docker:
```bash
export REACT_APP_API_SERVICE_URL=http://localhost:5004
docker-compose build
docker-compose up
```
Website will be at http://127.0.0.1:3007/app

If you'd like to set up the database:
```bash
docker-compose exec api python manage.py db upgrade
docker-compose exec api python manage.py seed_db
```