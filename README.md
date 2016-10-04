# viz-app
[![Build Status](http://deploy.graphistry.com/buildStatus/icon?job=viz-app)](http://deploy.graphistry.com/job/viz-app/)

Graphistry's cloud-based visualization platform. Stream WebGL buffers into the browser powered by an Open-CL based layout simulation.

## Instructions

1. Ensure you have `pigz` installed on your system. Mac: `brew install pigz`, Linux: `sudo apt-get install pigz`.
2. Clone this repo.
3. Run `npm install` from this repo's root directory.
  - This will install all dependencies, and run the module's build scripts to compile our es6 to (es5) JavaScript, and LESS to CSS.
4. Run `npm run dev` to clean, compile, and start a webpack-dev-server that hot reloads modules for the client and server.

Requires node `v6.6.0` and npm `v3.10.3`.

## Commands

- `npm run dev` cleans `www/` and compiles all modules for development
- `npm run build` cleans `www/` and compiles all modules for production
- `npm run serve` starts the server in `www/`
- `npm run debug` starts the server in `www/` with `--inspect --debug` options
- `npm run watch` cleans `www/` and starts parallel development builds, and starts the viz-server with hot-module reloading
- `npm run fancy` cleans `www/`, starts parallel development builds with `webpack-dashboard`, and starts the viz-server with hot-module reloading

## Test Datasets

https://docs.google.com/spreadsheets/d/1ctmEoT-aWjJyCiAZ9kEvW_XE9-ydyMCTtL9wGZna58Q/edit#gid=0
