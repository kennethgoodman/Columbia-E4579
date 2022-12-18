# pull official base image
FROM python:3.10.3
# FROM --platform=linux/x86-64 python:3.10.3
# FROM --platform=macosx/arm64 python:3.10.3
# FROM python:3.10.7

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

# RUN pip install tensorflow_decision_forests
# # build tensorflow decision forests from source
# RUN git clone https://github.com/tensorflow/decision-forests.git
# RUN cd decision-forests
# RUN ./tools/test_bazel.sh


# add app
COPY . .

# add entrypoint.sh
COPY ./entrypoint.sh .
COPY seed_data .
RUN chmod +x /usr/src/app/entrypoint.sh
