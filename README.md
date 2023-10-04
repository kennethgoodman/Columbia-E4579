# Columbia-E4579

Fall 2023 Class At Columbia. Modern Recommendation Systems

### Setup

#### Install Docker

must have `docker` and `docker-compose` installed. You can go to [docker's website](https://docs.docker.com/get-docker/)
to download the latest version of `docker` (comes with compose). Once you install it, you'll need to click on the application to run it.

Running the docker app will add `docker` and `docker-compose` to your PATH so you can run those commands on your terminal

### Build, Create DB, Seed DB and Run:

To do everything at once, open up a terminal and run the following command:

```bash
./docker_compose_run_setup_db.sh
```

The website will be at http://127.0.0.1:3007/feed

You must wait to see "FULLY DONE INSTANTIATION USE THE APP" to finish, you can comment out before_first_request if you don't need it

If you want to see the API docs, you can go to http://localhost:5004/doc

You can use `control-c` to kill the local servers

If you don't want to re-seed the DB, you can:

docker-compose -f docker-compose.full_db.yaml up --build

If you get errors, you may need to install git lfs:

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

If you don't want to download git lfs, you can download directly [from github](https://github.com/kennethgoodman/Columbia-E4579/blob/main/services/db/02_rest.sql) or [drive](https://drive.google.com/file/d/1fG-dI0Y792R73MaMtaBwH0ZWPL8aUZEB/view?usp=drive_link) and place it in the right spot (services/db/02_rest.sql)

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

#### Only build one service

For instance, the api backend
```bash
docker-compose -f docker-compose.full_db.yaml up --build api
```

### Access the dev database

If you want to access the database in a mysql CLI, you can run the following command while `docker-compose up` is running:

```bash
docker-compose exec api-db mysql --password=mysql api_dev
```

### Misc Commands

#### Access a python shell instantiated with the app env

```bash
docker-compose exec api python manage.py shell
```

#### Accessing terminal of api backend

```bash
docker-compose exec api /bin/bash
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
