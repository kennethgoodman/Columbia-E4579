# Columbia-E4579

Fall 2022 Class At Columbia. Modern Recommendation Systems

### Setup

#### Install Docker

must have `docker` and `docker-compose` installed. You can go to [docker's website](https://docs.docker.com/get-docker/)
to download the latest version of `docker` (comes with compose). Once you install it, you'll need to click on the application to run it.

Running the docker app will add `docker` and `docker-compose` to your PATH so you can run those commands on your terminal

#### Setup terminals

Open up two terminals, in both of them run:

```bash
export REACT_APP_API_SERVICE_URL=http://localhost:5004
```

### Build:

In your first terminal, you can build and install all depedencies by running:

```bash
docker-compose build
```

### Run Locally

In your first terminal, after you are done building the app, you can run the following command to bring up the web server, backend and database locally

```bash
docker-compose up
```

Website will be at http://127.0.0.1:3007/feed

### Set up the database:

If this is your first time running the app (or if you made a change to the data models) you should run the following command:
_You should run this in your SECOND terminal_ as the first terminal will be running the servers

```bash
docker-compose exec api python manage.py recreate_db
```

This command will delete all data in the local database and recreate the tables

### Seed the db:

In your second terminal following the creation of the database you should run

```bash
docker-compose exec api python manage.py seed_db
```

To seed the database

### Access the dev database

If you want to access the database in a postgres CLI, you can run the following command while `docker-compose up` is running:

```bash
docker exec -it $(docker ps | grep columbia-e4579_api-db | awk '{print $1}') psql -U postgres -d api_dev
```

## Thanks

1. [testdrivenio](https://github.com/testdrivenio/flask-react-aws) for a template for docker + flask + react
2. [Aveek-Saha](https://github.com/Aveek-Saha/Movie-Script-Database) for their scripts to download movie scripts
3. [lexica.art](https://lexica.art) for their collection of prompts/images to use as inspiration
4. [Reddit](https://reddit.com) and all their users for allowing the ability to programmattically download user titles/prompts
5.
