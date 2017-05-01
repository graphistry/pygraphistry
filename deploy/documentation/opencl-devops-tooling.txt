% Tooling to Containerize an OpenCL Visual Analytics Platform
% Lee Butterman, Graphistry, Inc.
% May 2017

# Welcome!

## Tooling to containerize an OpenCL Visual Analytics Platform
- We made our production deploys much easier
- We would like to share our tooling with you

## this is the app lol

## Overview
- What is OpenCL
- What is an app in production
- What is a container
- What is current tooling
- What we have built
- How you can use it and help 😀

# Background!

## What is OpenCL
- multi-platform parallel computing paradigm
- runs on GPUs, _CPUs_, FPGAs, &c
- host vs device
- like CUDA, a general purpose language calls a driver to run kernels

## What is a container
- < what is docker lol >

## What is an app in production

- an app: mostly other people's code
    - lots of libraries for lots of non-GPU-oriented tech (S3, HTTP, JSON, BCrypt, ...)
- in production: under siege from The Internet
    - production is Serious Business
    - the app's core competence is tough enough without using weird databases
- deploy environment may not be internet-accessible / may not access the internet
    - deploy to a moon base, deploy to a local machine

## What is our type of app in production

- Javascript is popular and okay
- People are optimizing its V8 runtime
- Server-side node.js + node-opencl = a clear optimization path
    - single-process map()
    - multi-process clusterMap()
    - CPU-mode kernel.run()
    - GPU-mode kernel.run() 🚀

## What is our app in production

- GPU-accelerated JS on client and server
    - WebGL on client, OpenCL on server
- Stateless visualization
- Front door with awareness of a cluster of processes
    - Viz session sticks to a process on a device with enough space
- Cluster membership
- Blob storage


## Where is our app deployed

- multiple environments
    - secret moon base, `us-east1`, some new cloud
        - (part of the excitement of a startup is customer development!)
- tooling is rapidly evolving
- simplifying assumption to treasure: 1 box can serve prod
    - sufficiently high availability on 1 machine in EC2
    - scale up 50x from a g2.2xl to a p2.16xl, punt on clustered deployment

## Conservative requirements

- Configure a machine to run containers and have any necessary drivers (automated, ideally)
- Verify that the machine is compatible with our stack
- Wrap up an artifact in development, ship it out
- Potentially reuse this artifact to run in development

# Things that have solved our problems

## Automated box setup

- Ansible script installs docker, nvidia drivers, nvidia-docker.
- Other ways to do it: docker-machine + run a few commands.
    - Tooling is evolving fast in the past 18 months
- https://github.com/graphistry/infrastructure : nvidia-docker-host

## Validate that we can run node-opencl code

- cl.js convolution demo
- Exercises nvidia-docker, opencl driver loader, node opencl bindings
    - some machines do not have OpenCL support
- Bonus: provides timings of CPU & GPU image convolutions
    - Extra bonus: there is a cute red panda
- <url>

## Package an artifact

- Must be self-contained!
- Loadable versions of all containers used, in a .tar.gz
- OpenCL on CPU is a few hundred MB compressed; on GPU, almost 1GB compressed
    - This is not a nimble 10MB image based on Alpine Linux
- `docker-gc` becomes more important

## Run containers in dev

- Provides a quick-start guide to run the app
- Until the year of Linux on the desktop 🙏 Docker is in a VM and cannot connect to the GPU
- In production we only need 60 fps; small jobs on the CPU?
    - we have this luxury, versus large batch jobs (eg radiological deep learning)
- dockerhub: graphistry/js-and-cpu

# This is only the begining!

## Further directions

- more configurations! (FPGA devices! ARM hosts! Power8‽)
- tighter integration with Docker?
- Kubernetes!
- smaller images!

## Thank you!

### Lee Butterman, lsb@graphistry.com

#### Questions? Comments? Discussions?

