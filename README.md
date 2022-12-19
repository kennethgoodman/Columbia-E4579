# Columbia-E4579

Fall 2022 Class At Columbia. Modern Recommendation Systems

### Setup

#### Install Docker

must have `docker` and `docker-compose` installed. You can go to [docker's website](https://docs.docker.com/get-docker/)
to download the latest version of `docker` (comes with compose). Once you install it, you'll need to click on the application to run it.

Running the docker app will add `docker` and `docker-compose` to your PATH so you can run those commands on your terminal

### Build, Create DB, Seed DB and Run:

To do everything at once, open up a terminal and run the following command:

_WARNING_ running this command WILL delete all data in the local database, such as local likes/content views. It will then reseed the DB

Use this command if you want a clean database back to the original seed (see seed_data folder).

```bash
docker-compose up --build --force-recreate
```

You only need to seed the DB once, afterwards you can:

```bash
docker-compose -f docker-compose.yml -f docker-compose.override.no.recreate.yaml up
```

The website will be at http://127.0.0.1:3007/feed

If you want to see the API docs, you can go to http://localhost:5004/doc

You can use `control-c` to kill the local servers

If you don't want to re-seed the DB, you can:

If you want to do things separately there are 3 commands:

#### Running With Engagement Dump (Takes Longer)

If you're running mac and have HomeBrew installed, then you can run brew install git-lfs

You NEED to download `git lfs`. Go to https://git-lfs.github.com/ and download/install git lfs.

Then clone the repo (AFTER you have installed git lfs)

Clone repo
```bash
git clone https://github.com/kennethgoodman/Columbia-E4579.git
```

Then cd into the directory and run git lfs
```bash
git lfs install && git lfs pull
```

The first time to build the database

```bash
docker-compose -f docker-compose.full_db.yml up --build --force-recreate --renew-anon-volumes
```

Afterwards you can run

```bash
docker-compose -f docker-compose.full_db.yaml up
```

#### Bring Down Containers

```bash
docker-compose down
```

#### Build containers (without a cache)

```bash
docker-compose build --no-cache
```

#### Bring up the containers

```bash
docker-compose up
```

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

#### Accessing terminal of api backend

```bash
docker exec -it $(docker ps | grep "/usr/src/app/entryp…" | awk '{print $1}') /bin/bash
```

#### Recreating DB

```bash
docker-compose exec api python manage.py recreate_db
```

#### Seed DB

```bash
docker-compose exec api python manage.py seed_db
```

#### Deploy To AWS

##### Build Frontend

Run the scripts/deploy_frontend.sh

##### Docker EC2

```bash
docker-compose -f docker-compose.prod.yml up --build
```

## Thanks

1. [testdrivenio](https://github.com/testdrivenio/flask-react-aws) for a template for docker + flask + react
2. [Aveek-Saha](https://github.com/Aveek-Saha/Movie-Script-Database) for their scripts to download movie scripts
3. [lexica.art](https://lexica.art) for their collection of prompts/images to use as inspiration
4. [Reddit](https://reddit.com) and all their users for allowing the ability to programmattically download user titles/prompts
5. [@SamuelSacco](https://github.com/SamuelSacco) for their work on the frontend
