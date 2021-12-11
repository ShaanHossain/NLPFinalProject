FROM tensorflow/tensorflow:latest-gpu-jupyter

WORKDIR /tf

COPY . /tf

RUN ls -la /tf

RUN pip install -r requirements_docker.txt
