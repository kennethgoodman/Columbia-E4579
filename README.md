# Columbia-E4579
Fall 2022 Class At Columbia. Modern Recommendation Systems

### Build:
must have docker and docker-compose installed.

```bash
export REACT_APP_API_SERVICE_URL=http://localhost:5004
docker-compose build
```

### Run Locally
```bash
docker-compose up
```
Website will be at http://127.0.0.1:3007/feed

### Set up the database:
```bash
docker-compose exec api python manage.py recreate_db
```

### Seed the db:
```bash
docker-compose exec api python manage.py seed_db
```

### Access the dev database 
while `docker-compose up` is running:
```bash
docker exec -it $(docker ps | grep columbia-e4579_api-db | awk '{print $1}') psql -U postgres -d api_dev
```