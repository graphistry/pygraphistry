# viz-app
[![Build Status](http://deploy.graphistry.com/buildStatus/icon?job=viz-app)](http://deploy.graphistry.com/job/viz-app/)

Graphistry's cloud-based visualization platform. Stream WebGL buffers into the browser powered by an Open-CL based layout simulation.

## Instructions

1. Ensure you have `pigz` installed on your system. Mac: `brew install pigz`, Linux: `sudo apt-get install pigz`.
2. Clone this repo.
3. Run `npm install` from this repo's root directory.
  - This will install all dependencies, and run the module's build scripts to compile our es6 to (es5) JavaScript, and LESS to CSS.
4. Run `npm run dev` to clean, compile, and start a webpack-dev-server that hot reloads modules for the client and server.

Requires node `v6.1.0` and npm `v3.9.3`.

## Commands

- `npm run build` cleans `www/` and compiles all modules for production
- `npm start` cleans `www/`, compiles all modules for production, and starts the viz-server
- `npm run dev` cleans `www/`, compiles all modules for development, and starts a webpack-dev-server that watches source files and hot-reloads the viz-client and viz-server

## Test Datasets

https://docs.google.com/spreadsheets/d/1ctmEoT-aWjJyCiAZ9kEvW_XE9-ydyMCTtL9wGZna58Q/edit#gid=0
