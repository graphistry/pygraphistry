# Table of Contents
1. Client
2. Notebook Integration
3. Server: Cloud Option
4. Server: On-Premise Option

# Client

The client requires no installation beyond a modern browser: it is a standards-compliant web app. All modern operating systems with modern graphics drivers are supported (Windows, OS X, Linux).

## Web Browser
* Software: Chrome/Firefox/IE with WebGL and websockets
* GPU: most discrete cards from 2012+, such as Nvidia GeForce, with 1GB+ GPU RAM, or a high-quality integrated card
* CPU: 8GB+ RAM (3GB+ free), recommended multicore

Test your client hardware with demos of different sizes at github.com/graphistry/pygraphistry

### Sample successful configuration we use internally

* **Discrete**: MacBook Pro (Retina 2012)
  * 2.6 GHz "Ivy Bridge" Intel Core i7-3720QM, 8 GB Ram
  * Nvidia GeForce GT 650M with 1GB RAM

* **Integrated**: MacBook Pro (Retina, Mid 2014)
  *  3 GHz "Haswell" Intel Core i7-4578U, 8 GB Ram
  *  Intel Iris Graphics 5100 1536 MB

### Even thinner clients

Server offloading allows the possibility of thinner clients, including dropping the WebGL requirement. Please contact for discussion of timelines and impact on serverside multitenancy.


# Notebook Integration

We actively support IPython/Juypter, including Hub for team settings. Prototypes successfully ran in other notebooks such as Databricks, Zeppelin, and Wakari (Continuum). If needed, further notebook environments can be added to the official support set.

For python use, we recommend also installing `pyspark`, `pandas`, `networkx`, and `igraph`. See github.com/graphistry/pygraphistry for more information.


# Server: Cloud Option

* We support AWS (including GovCloud)
* We can host a shared or private instance
* Contact about launching and maintaining an AMI (or multiple AMIs) in your system. Depending on cluster size, we recommend one or more G2 instances, and potentially, a dedicated bastion proxy instance.

# Server: On-Premise Option

## Hardware: one or more OpenCL servers

**Required** A Linux-based server with OpenCL 1.2+ . Contact before using non-Nvidia GPUs, or if purely multicore/SIMD is mandatory.

**Recommended** Discrete Nvidia GPU (K40/K80), with multicore.

## Operating System

**Required** A Linux-based server with OpenCL 1.2+ .

**Recommended** Ubuntu 14.04 LTR "Trusty"  (AWS: HVM as the AMI)

Installation from a non-graphistry base image may require root access when setting up Nvidia GPU drivers and nginx.

## User-Level Packages

We use an ansible script (and apt, npm) to install the following:

### System

We can provide current version numbers at time of installation.

ansible, glfw, mongo

### apt

We can provide current version numbers at time of installation.


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

We can provide current version numbers at time of installation.

**normal dependencies**

    bunyan, chalk, config, fs-tools, q, segfault-handler, underscore, debug, lodash, yargs, common, protobufjs, sprintf-js, socket.io-client, body-parser, slack-write, express, orchestrate, express3-handlebars, handlebars-paginate, rx, ace-builds, color, colorbrewer, dateformat, dijkstra, dns, express-http-proxy, gl-matrix, less, mongodb, node.extend, node-pigz, node-webgl, node-opencl, request, socket.io, supertest, watch-less, xhr2, asty, backbone, backbone.paginator, backgrid, backgrid-paginator, backgrid-filter, bootstrap-slider, brace, brfs, d3, esprima-fb, grunt, grunt-browserify, grunt-cli, grunt-contrib-clean, grunt-contrib-jshint, grunt-contrib-watch, grunt-exorcise, grunt-peg, handlebars, immutable, numeric, pegjs-util, compression, aws-sdk, heap, iconv-lite, bindings, chai, get-pixels, nan, png-js, node-gyp,

**optional development dependencies**

    grunt-contrib-clean, grunt-jsduck, grunt-contrib-uglify, grunt-cli, grunt, grunt-recess, grunt-template-jasmine-istanbul, grunt-contrib-jasmine, jasmine-node, StreamGL, karma, pegjs, mocha, sinon, should, xml2js, JSONStream, q, jschardet, chalk, underscore



