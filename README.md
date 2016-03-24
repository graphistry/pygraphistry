# graph-viz
[![Build Status](http://deploy.graphistry.com/buildStatus/icon?job=graph-viz)](http://deploy.graphistry.com/job/graph-viz/)

Graph visualization for Graphistry's cloud-based visualization platform.

This module is primarily intended to be loaded by [viz-server](https://github.com/graphistry/viz-server), where it implements a OpenCL-based graph layout simulation and [StreamGL](https://github.com/graphistry/StreamGL)-compatible data generation. 

This module also has basic support for being run as a stand-alone app, strictly for development/testing purposes. 

## Instructions

1. Ensure you have `pigz` installed on your system. Mac: `brew install pigz`, Linux: `sudo apt-get install pigz`.
2. Clone this repo.
3. Run `npm install` from this repo's root directory.
  - This will install all dependencies, and run the module's build scripts to compile our es6 to (es5) JavaScript, and LESS to CSS.
4. After making any changes to the source code, run `npm build` to recompile the source. 
  - **If you fail to do this, your code changes won't have any effect on the running code.**
5. Alternatively, instead of step 4, you can run `npm watch`.
  - While this command is running, it watches for any changes made to the source files, and automatically recompiles the code when a change is detected.


## Standalone Development Server

This modules implements a bare-bones equivalent of [viz-server](https://github.com/graphistry/viz-server), to provide a HTTP/WebSocket interface to the graph visualization. This is not meant to be run in production *ever*, and lacks many core features compared to viz-server.

> It is reccomended that, even for local development, you run viz-server, which will load graph-viz, instead of running graph-viz directly.

If you still insist on running graph-viz directly, run `npm start` in this repo's root directory.
