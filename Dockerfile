FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update
RUN apt-get install -y software-properties-common git screen nano htop curl unzip wget sudo
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
RUN apt-get install -y python3.9 python3.9-distutils python3-pip

WORKDIR /app
COPY requirements.txt requirements.txt
RUN python3.9 -m pip install -r requirements.txt

COPY . /app/regvd/

WORKDIR /app/regvd/code
CMD ["sleep","infinity"]
