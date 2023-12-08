# syntax=docker/dockerfile:1
FROM python:3.8.17-slim-bullseye
LABEL GPUS=0 GPU_MEMORY=0
LABEL description="Genetic Algorithm AIM Example."
COPY . /app
ENV DEBIAN_FRONTEND=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    curl \
    ffmpeg \
    ca-certificates && \
  rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install -r /app/requirements.txt

COPY . .

EXPOSE 4002/tcp

CMD python3 genetic_server.py