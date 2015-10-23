# Graphistry Developer-Mode Tarball

Developer mode for evaluating same machine use. Launches 'localhost:3000' (and 'localhost:10001...10010').  

In contrast, the production environment is cleaner and more flexible, supporting redeploys, multinode clustering, logging, automatic restart, file saving, etc.

## Node Version

Tested with:
   ubuntu@14.04.03
   node@0.10.40
   npm@2.14.4

## Dependencies

1. Ensure GPU driver, CUDA, OpenCL, and OpenCL headers are already installed: [http://vasir.net/blog/opencl/installing-cuda-opencl-pyopencl-on-aws-ec2](http://vasir.net/blog/opencl/installing-cuda-opencl-pyopencl-on-aws-ec2)

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

[http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/lesmiserables.vgraph.gz&scene=default&info=true&play=10000](http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/lesmiserables.vgraph.gz&scene=default&info=true&play=10000)

## Test Notebook

Install IPython/Jupyter and PyGraphistry using instructions from [https://github.com/graphistry/pygraphistry](https://github.com/graphistry/pygraphistry).

- Warning: the notebook must be open using HTTP, not HTTPS. For instance, `http://localhost:8888`.

To link the notebook to the visualization server, use the following `register` call at the beginning of each notebook.

```python
import graphistry
graphistry.register('nokey', 'localhost:3000', protocol='http')
```

If the IPython and the visualization server run on different machines, replace `localhost` by the ip of the visualization server.



