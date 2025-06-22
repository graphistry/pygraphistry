#!/bin/bash

docker run -it --rm \
  --name notebook \
  -e PYTHONPATH="/opt/py_env:/pygraphistry" \
  -e PIP_TARGET="/opt/py_env" \
  -e USERNAME=leotest2 \
  -e GRAPHISTRY_PASSWORD=accountaccount \
  -p 8888:8888 \
  -v /home/lmeyerov/Work/pygraphistry:/pygraphistry \
  -v /home/lmeyerov/Work/pygraphistry/demos:/home/graphistry/demos \
  graphistry/jupyter-notebook:v2.41.17-11.8

