# pull official base image
FROM python:3.10.3-slim-buster

# set working directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update \
  && apt-get -y install netcat gcc postgresql \
  && apt-get clean

# install postgres
RUN apt update -y && apt install -y build-essential libpq-dev
RUN pip install psycopg2-binary --no-binary psycopg2-binary

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
COPY ./requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# add app
COPY . .

# add entrypoint.sh
COPY ./entrypoint.sh .
COPY seed_data .
RUN chmod +x /usr/src/app/entrypoint.sh
