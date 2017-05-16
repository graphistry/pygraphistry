% Tooling to Containerize an OpenCL Visual Analytics Platform
% Lee Butterman, Graphistry, Inc.
% May 2017

## Graphistry is using GPUs to power the future of visual analytics

### >100⨉ data in the first client⟷cloud GPU visual analytics platform: see all known proteins

<iframe width=100% height=75% src="https://labs.graphistry.com/graph/graph.html?dataset=Biogrid&splashAfter=0"></iframe>

## Every known protein in the BioGrid database at a glance

![](./biogrid-tda.png)

## Real--time zoom in to see detail

![](./biogrid-zoom.png)

## Inspection over all columns of data

![](./biogrid-datatable.png)

## Scrub a histogram to see different clusters

![](./biogrid-histo.png)

## How to build/deploy a web app with a GPU-accelerated HTTP loop?

### Ideal: change OpenCL kernels, 1-click deploy to environments in minutes!

## Plan

- Reproducible builds: artifact-based deploys
- Host management: GPU drivers et cetera on the box
- Validation: minimize GPU surprises
- Fallback: multicore via CPU

## Problem: reproducible builds

Deploy artifact to staging, production, a customer’s air gapped network

Easily re-deploy old build

Docker is popular and has a huge container ecosystem

## Problem: make Docker talk to GPU

`nvidia-docker`: wraps the Docker CLI

Need drivers on disk

Customers on Ubuntu, RHEL, and more

Install Docker, nvidia-docker, drivers on RHEL/Ubuntu ⇒ our Ansible script!

`https://github.com/graphistry/infrastructure/tree/master/nvidia-docker-host`

`nvidia-docker` 2.0: native orchestration docker-swarm/kubernetes/mesos

## Add nodejs to GPU-accelerated containers

We need company-wide base containers of app runtime + OpenCL drivers

Pull from dockerhub: `graphistry/`{`cpu`,`gpu`}`-base`, `graphistry/js-and-`{`cpu`,`gpu`}

## Auto-test GPU assumptions!

Insufficient: `nvidia-smi` alone

Better: `clinfo`, testing `node-opencl`, wide coverage

Use our library `cl.js` to do a simple image convolution

Pull from dockerhub: `graphistry/cljs`

<iframe width=100% height=50% src="http://52.9.9.187:3001/?mode=opencl"></iframe>

## CPU mode for the full app is a great idea

`nvidia-docker` only supports Linux

Many developers are not using Linux natively

Sufficient performance, much less cost

Pull from docker hub: `graphistry/js-and-cpu`

## Thank you!

Build apps and tests on top of GPU and CPU OpenCL containers, package an artifact for a deploy, set up a nvidia-docker host to run the artifact, go from 0 to new machine running new code in half an hour

`https://github.com/graphistry/infrastructure`

### Lee Butterman

### lsb@graphistry.com
