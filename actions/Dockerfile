FROM python:3.9-slim-buster

RUN apt update \
    && apt upgrade \
    && apt install python3-dev default-libmysqlclient-dev build-essential -y

RUN mkdir /gretel && mkdir /gretel/data
WORKDIR /gretel

COPY requirements.txt /gretel/
COPY *.py /gretel/

RUN pip install -r requirements.txt

CMD ["python", "run.py"]
