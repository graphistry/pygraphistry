# graph-viz
[![Build Status](http://deploy.graphistry.com/buildStatus/icon?job=graph-viz)](http://deploy.graphistry.com/job/graph-viz/)

Graph visualization for Graphistry's cloud-based visualization platform.
[![Build Status](http://deploy.graphistry.com/buildStatus/icon?job=StreamGL)](http://deploy.graphistry.com/job/StreamGL/)

Stream WebGL buffers into the browser from a remote server
This module is primarily intended to be loaded by [viz-server](https://github.com/graphistry/viz-server), where it implements a OpenCL-based graph layout simulation and [StreamGL](https://github.com/graphistry/StreamGL)-compatible data generation. 


## Instructions

1. Ensure you have `pigz` installed on your system. Mac: `brew install pigz`, Linux: `sudo apt-get install pigz`.
2. Clone this repo.
3. Run `npm install` from this repo's root directory.
  - This will install all dependencies, and run the module's build scripts to compile our es6 to (es5) JavaScript, and LESS to CSS.
4. After making any changes to the source code, run `npm build` to recompile the source. 
  - **If you fail to do this, your code changes won't have any effect on the running code.**
5. Alternatively, instead of step 4, you can run `npm watch`.
  - While this command is running, it watches for any changes made to the source files, and automatically recompiles the code when a change is detected.
Requires node `v0.10.41` and npm `v2.14.0`.

- `npm run build` compiles for production
- `npm run build-dev` compiles for development
- `npm run start` compiles for production and starts a file watcher
- `npm run start-dev` compiles for development and starts a file watcher
- `npm run dev` is an alias for `npm run start-dev`



## Test Datasets

https://docs.google.com/spreadsheets/d/1ctmEoT-aWjJyCiAZ9kEvW_XE9-ydyMCTtL9wGZna58Q/edit#gid=0
