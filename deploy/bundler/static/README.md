# Graphistry Developer-Mode Tarball

Developer mode for evaluating same machine use. Launches 'localhost:3000' (and 'localhost:10001...10010').  

In contrast, the production environment supports redeploys, multinode clustering, logging, automatic restart, file saving, etc.

## Dependencies

1. Ensure OpenCL GPU driver is already installed.
2. `sudo apt-get install pigz`

## Install

```
./install.sh
```

## Run Server

```
cd bin
./server.sh
```

## Stop Server

```
killall node
```
