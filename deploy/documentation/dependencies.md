# Client

## Web Browser

* Software: Chrome/Firefox/IE with WebGL and websockets
* CPU: 8GB+ RAM (3GB+ free), recommended multicore
* GPU: discrete, such as Nvidia GeForce, with 1GB+ GPU RAM

Test your client hardware with demos of different sizes at github.com/graphistry/pygraphistry

### Sample successful configurations:

We use internally:

* Discrete: 2012 MacBook Pro, 2.6GHz Intel Core i7, 8GB RAM, Nvidia GeForce GT 650M with 1GB RAM

* Integrated: 2014 MacBook Pro, 3 GHz Intel Core i7 8 GB, Intel Iris 1536 MB

### Even thinner clients

Server offloading allows the possibility of thinner clients, including dropping the WebGL requirement. Please contact for discussion of timelines and impact on serverside multitenancy.


## Notebooks

We actively support IPython/Juypter, including Hub for team settings. Prototypes successfully ran in other notebooks such as Databricks, Zeppelin, and Wakari (Continuum). If needed, further notebook environments can be added to the official support set.

For python use, we recommend also installing `pyspark`, `pandas`, `networkx`, and `igraph`. See github.com/graphistry/pygraphistry for more information.


# Server

## Hardware: one or more OpenCL servers

**Required** A Linux-based server with OpenCL 1.2+ . Contact before using non-Nvidia GPUs, or if purely multicore/SIMD is mandatory. On AWS, G2/G8 instances.

**Recommended** Discrete Nvidia GPU (K40/K80), with multicore.

## Operating System

**Required** A Linux-based server with OpenCL 1.2+ .
**Recommended** Ubuntu 14.04 LTR "Trusty"  (AWS: HVM as the AMI)

Installation from a non-graphistry base image may require root access when setting up Nvidia GPU drivers and nginx.

## User-Level Packages

### System
ansible, glfw, AntTweakBar, mongo

### apt

    supervisor, nginx,

    build-essential, curl, git, logrotate, vim, ack-grep, python-pip, unattended-upgrades, python-psutil, ntp, pkg-config, acl, schedtool

    60nvidia
    cuda-repo-ubuntu1404_7.0-28_amd64.deb
    cuda-7-0
    linux-image-extra-virtual, mesa-utils, libxrandr-dev, libxcursor-dev, libglew-dev,
    libglu1-mesa-dev, libfreeimage-dev, libxinerama-dev, xinput, libxi-dev, cmake, unzip

    libkrb5-dev, nodejs, node-gyp

    pigz

### npm
(our repos are private on github, can retarget)

**normal dependencies**

    bunyan, chalk, config, fs-tools, q, segfault-handler, underscore, debug, lodash, yargs, common, protobufjs, sprintf-js, socket.io-client, body-parser, slack-write, express, orchestrate, express3-handlebars, handlebars-paginate, rx, graph-viz, ace-builds, color, colorbrewer, dateformat, dijkstra, dns, express-http-proxy, gl-matrix, less, mongodb, node.extend, node-pigz, node-webgl, node-opencl, request, socket.io, supertest, watch-less, xhr2, asty, backbone, backbone.paginator, backgrid, backgrid-paginator, backgrid-filter, bootstrap-slider, brace, brfs, d3, esprima-fb, grunt, grunt-browserify, grunt-cli, grunt-contrib-clean, grunt-contrib-jshint, grunt-contrib-watch, grunt-exorcise, grunt-peg, handlebars, immutable, numeric, pegjs-util, StreamGL, compression, splunk-viz, uber-viz, aws-sdk, heap, iconv-lite, bindings, chai, get-pixels, nan, png-js, node-gyp, etl-worker

**optional development dependencies**

    grunt-contrib-clean, grunt-jsduck, grunt-contrib-uglify, grunt-cli, grunt, grunt-recess, grunt-template-jasmine-istanbul, grunt-contrib-jasmine, jasmine-node, StreamGL, karma, pegjs, mocha, sinon, should, xml2js, JSONStream, q, jschardet, chalk, underscore



