#!/usr/bin/env bash

apt-get install -y python python-pip python-dev

pip install subprocess32 gradescope-utils

apt-get update

apt-get install -y software-properties-common python-software-properties

add-apt-repository ppa:deadsnakes/ppa

apt-get update

apt update

apt install -y python3.6

apt install -y python3.6-dev

apt install -y python3.6-venv

wget https://bootstrap.pypa.io/get-pip.py

python3.6 get-pip.py

pip install numpy
