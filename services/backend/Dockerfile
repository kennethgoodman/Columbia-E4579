# pull official base image
FROM python:3.10.3

# set working directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_SETTINGS src.config.DevelopmentConfig

RUN pip install mysql-connector-python

# install system dependencies
RUN apt-get update \
  && apt-get -y install gnupg netcat g++ gcc build-essential python-dev \
  && apt-get clean

# install mysql
RUN pip install mysqlclient

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
COPY ./requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# install mrpt for approximate NN
RUN pip install git+https://github.com/vioshyvo/mrpt/

# add app
COPY . .

# add entrypoint.sh
COPY ./entrypoint.sh .
COPY seed_data .
RUN chmod +x /usr/src/app/entrypoint.sh
