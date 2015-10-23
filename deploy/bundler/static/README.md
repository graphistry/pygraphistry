# Graphistry Developer-Mode Tarball

Developer mode for evaluating same machine use. Launches 'localhost:3000' (and 'localhost:10001...10010').  

In contrast, the production environment supports redeploys, multinode clustering, logging, automatic restart, file saving, etc.

## Node Version

Tested with:
   ubuntu@14.04.03
   node@0.10.40
   npm@3.3.9

## Dependencies

1. Ensure GPU driver, CUDA, OpenCL, and OpenCL header is already installed: http://vasir.net/blog/opencl/installing-cuda-opencl-pyopencl-on-aws-ec2
2. `sudo apt-get install pigz`

## Install

```
./install.sh
```

## Run Server

```
cd bin
./server.sh
```

## Stop Server

```
killall node
```
