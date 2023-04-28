FROM python:3.9-slim-buster as builder

RUN apt update \
    && apt upgrade \
    && apt install python3-dev default-libmysqlclient-dev build-essential -y

RUN mkdir /gretel && mkdir /gretel/data
WORKDIR /gretel

COPY requirements.txt /gretel/
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.9-slim-buster
COPY --from=builder /root/.local /root/.local
COPY --from=builder /usr/lib/x86_64-linux-gnu/*.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/include /usr/include

COPY *.py /gretel/

WORKDIR /gretel

ENV PATH=/root/.local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH


CMD ["python", "run.py"]
