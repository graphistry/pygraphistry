# Graphistry Developer-Mode Tarball

**Developer mode** for evaluating same machine use. Launches `localhost:3000` , with background workers `localhost:10001...10010` and corresponding log files.

**Contact for production mode**, which is cleaner and more flexible. It supports redeploys, secure sessions, https, reverse proxying, multinode clustering, logging, automatic restart, file saving, etc.

## OS/Node Version

Tested with:

*   ubuntu@14.04.03
*   node@0.10.40
*   npm@2.14.4

## External Dependencies

1. Ensure the GPU driver, CUDA, OpenCL, and OpenCL headers are already installed: [http://vasir.net/blog/opencl/installing-cuda-opencl-pyopencl-on-aws-ec2](http://vasir.net/blog/opencl/installing-cuda-opencl-pyopencl-on-aws-ec2)

  *Note:* in the above script, the python dependencies are unnecessary (half of step 3.0 and all of step 4.0)

2. `sudo apt-get install pigz`

## Install

After satisfying the above dependencies:

```
tar -xhvfj graphistry-2015-10-23.ubuntu14.04.x86-64.tar.bz2 
```

*Note:* The "`h`" option is necessary for preserving symbolic link structure.

## Run Server

**About:** Starts the main server and several workers. 

* *Warning*: unlike production, the developer mode will reuse workers across visualizations, and thus leak state. Likewise, unlike production mode, there is little fault tolerance.

* *Warning*: runs `killall node`, which will stop any current NodeJS processes

####The Command

```
cd bin
./start.sh
```

####Expected output

```
bin$ ./start.sh 
{"HTTP_LISTEN_ADDRESS": "0.0.0.0", "VIZ_LISTEN_PORTS": [10001,10002,10003,10004,10005,10006,10007,10008,10009,10010]}
{"VIZ_LISTEN_ADDRESS": "0.0.0.0", "VIZ_LISTEN_PORT": 10001}
{"VIZ_LISTEN_ADDRESS": "0.0.0.0", "VIZ_LISTEN_PORT": 10002}
{"VIZ_LISTEN_ADDRESS": "0.0.0.0", "VIZ_LISTEN_PORT": 10003}
{"VIZ_LISTEN_ADDRESS": "0.0.0.0", "VIZ_LISTEN_PORT": 10004}
{"VIZ_LISTEN_ADDRESS": "0.0.0.0", "VIZ_LISTEN_PORT": 10005}
{"VIZ_LISTEN_ADDRESS": "0.0.0.0", "VIZ_LISTEN_PORT": 10006}
{"VIZ_LISTEN_ADDRESS": "0.0.0.0", "VIZ_LISTEN_PORT": 10007}
{"VIZ_LISTEN_ADDRESS": "0.0.0.0", "VIZ_LISTEN_PORT": 10008}
{"VIZ_LISTEN_ADDRESS": "0.0.0.0", "VIZ_LISTEN_PORT": 10009}
{"VIZ_LISTEN_ADDRESS": "0.0.0.0", "VIZ_LISTEN_PORT": 10010}
node: no process found
```


## Stop Server

```
killall node
```

## Debug

#### Browser

Open the developer JavaScript console to learn the ID of the backend worker and see any clientside error messages.

#### Server

Open `viz-server/


## Test/Use

### Premade Visualizations

See folder `/central/assets/datasets` for sample visualization names. Plug the right base IP into a URL like:

**Small**

* Les Miserables: [http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/Miserables&scene=default&info=true&play=10000](http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/Miserables&scene=default&info=true&play=10000)

* Facebook 1.5 Friends-of-Friends Graph [http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/Facebook&scene=default&info=true&play=10000](http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/Facebook&scene=default&info=true&play=10000)

**Medium**

* Silkroad's Tainted Bitcoins (Sept): [http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/BitcoinTaintGraph&scene=default&info=true&play=10000](http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/BitcoinTaintGraph&scene=default&info=true&play=10000)

* Port Scan (Priority 10): [http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/PortScan&scene=default&info=true&play=10000](http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/PortScan&scene=default&info=true&play=10000)

* Port Scan (Escalation) [http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/PortScanEscalation&scene=default&info=true&play=10000](http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/PortScanEscalation&scene=default&info=true&play=10000)

* Twitter Botnet [http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/Twitter&scene=default&info=true&play=10000](http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/Twitter&scene=default&info=true&play=10000)


**Large**

* Protein Network (Biogrid) [http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/Biogrid&scene=default&info=true&play=10000](http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/Biogrid&scene=default&info=true&play=10000)

* Netflows [http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/NetflowLarge&scene=default&info=true&play=10000](http://54.149.52.47:3000/graph/graph.html?dataset=http://localhost:3000/datasets/NetflowLarge&scene=default&info=true&play=10000)


### IPython/Jupyter Notebook

Install IPython/Jupyter and PyGraphistry using instructions from [https://github.com/graphistry/pygraphistry](https://github.com/graphistry/pygraphistry).

- Warning: in developer mode, notebooks must be open using HTTP, not HTTPS. For instance, `http://localhost:8888`.

To link the notebook to the visualization server, use the following `register` call at the beginning of each notebook.

```python
import graphistry
graphistry.register('e41c3f48e380c809107abdac54a37207a3252c271c79ece0865a44f40cc6ebb8', 'localhost:3000', protocol='http')
```

If the IPython and the visualization server run on different machines, replace `localhost` by the IP of the visualization server.



