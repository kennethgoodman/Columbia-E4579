# Columbia-E4579

Fall 2022 Class At Columbia. Modern Recommendation Systems

### Setup

#### Install Docker

must have `docker` and `docker-compose` installed. You can go to [docker's website](https://docs.docker.com/get-docker/)
to download the latest version of `docker` (comes with compose). Once you install it, you'll need to click on the application to run it.

Running the docker app will add `docker` and `docker-compose` to your PATH so you can run those commands on your terminal

### Build, Create DB, Seed DB and Run:

To do everything at once, open up a terminal and run the following command:

_WARNING_ running this command WILL delete all data in the local database

```bash
docker-compose up --build --force-recreate
```

The website will be at http://127.0.0.1:3007/feed

You can use `control-c` to kill the local servers

### Access the dev database

If you want to access the database in a mysql CLI, you can run the following command while `docker-compose up` is running:

```bash
docker exec -it $(docker ps | grep columbia-e4579_api-db | awk '{print $1}') mysql --password=mysql api_dev
```

### Misc Commands

#### Access a python shell instantiated with the app env

```bash
docker-compose exec api python manage.py shell
```

#### Recreating DB

```bash
docker-compose exec api python manage.py recreate_db
```

#### Seed DB

```bash
docker-compose exec api python manage.py seed_db
```

## Thanks

1. [testdrivenio](https://github.com/testdrivenio/flask-react-aws) for a template for docker + flask + react
2. [Aveek-Saha](https://github.com/Aveek-Saha/Movie-Script-Database) for their scripts to download movie scripts
3. [lexica.art](https://lexica.art) for their collection of prompts/images to use as inspiration
4. [Reddit](https://reddit.com) and all their users for allowing the ability to programmattically download user titles/prompts
5.
