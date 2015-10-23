# Graphistry Developer-Mode Tarball

Developer mode for evaluating same machine use. Launches 'localhost:3000' (and 'localhost:10001...10010').  

In contrast, the production environment is cleaner and more flexible, supporting redeploys, multinode clustering, logging, automatic restart, file saving, etc.

## Node Version

Tested with:
   ubuntu@14.04.03
   node@0.10.40
   npm@3.3.9

## Dependencies

1. Ensure GPU driver, CUDA, OpenCL, and OpenCL headers are already installed: http://vasir.net/blog/opencl/installing-cuda-opencl-pyopencl-on-aws-ec2

  Note: in the above script, the python dependencies are unnecessary (half of step 3.0 and all of step 4.0)

2. `sudo apt-get install pigz`

## Install

```
tar -xfhj graphistry-2015-10-23.ubuntu14.04.x86-64.tar.bz2 
```

Note the "h" option to preserve symbolic link structure.

## Run Server

Starts the main server and several workers. 

* Warning: unlike production, the developer mode will reuse workers across visualizations, and thus leak state.

* Warning: runs `killall node`, which will stop any current NodeJS processes

```
cd bin
./start.sh
```

## Stop Server

```
killall node
```

## Test Server

See folder /central/assets/datasets for sample visualization names. Plug into a URL like:

http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/lesmiserables.vgraph.gz&scene=default&info=true&play=10000

## Test Notebook


