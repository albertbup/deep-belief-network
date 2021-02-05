FROM python:3.6-slim

RUN mkdir /code
WORKDIR /code

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt