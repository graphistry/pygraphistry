# Installation Instructions

These are instructions to get a single-node setup of Graphistry up and running, on an Ubuntu/RedHat box that you have provisioned. You should have received this inside a package of:

  * *instructions.md* (these instructions)
  * *certs.txt* (how to generate certs)
  * *containers.lxc.gz* (the Docker containers with the relevant code)
  * *load.sh* (loads containers for use)
  * *launch.sh* (run the server)

### What Will We Be Doing?

First, we will be preparing a machine to run Docker containers that can take advantage of the machine's GPU. This is a one-time task.
Second, we will be loading a few containers onto that machine that, in the ensemble, run Graphistry.

### How Long Should This Take?

Preparing a machine to run Docker containers should take around an hour (provisioning, finding a new version of Ubuntu/RedHat, running the scripts).
Loading new versions of the containers onto the machine should take a minute or two, and the launch script should take ten seconds or so.

## 1 Prerequisites And Supported Hardware

You will need a box:

 * connected to the Internet,
 * connected to a box running an Ansible playbook, also connected to the Internet.
 * with a GPU that can run Cuda 7.5,
 * with the Docker engine installed,
 * and optionally with the nvidia-docker standalone installation.

Currently, the only supported and tested configuration are

 * Ubuntu 14.04 Trusty LTS or RedHat Enterprise Linux (RHEL) 7.3
 * NVIDIA Grid K2 (g2.2xlarge on AWS EC2) or NVIDIA Tesla K40/K80.

## 2 Prepare The Machine


#### Set up Nvidia-Docker With Our Process (Option A)

1. Checkout the infrastructure repository at [https://github.com/graphistry/infrastructure/](https://github.com/graphistry/infrastructure/).
2. Run through the setup at [https://github.com/graphistry/infrastructure/tree/master/nvidia-docker-rhel](https://github.com/graphistry/infrastructure/tree/master/nvidia-docker-host).

If the script runs through, your machine will have passed a test of using Docker to run GPU computation (matrix multiplication in this case).

### Set up Nvidia-Docker Manually (Option B)

1. (Use your process to set up nvidia-docker.)
2. Verify that your process succeeds by pulling down the `nvidia/cuda:7.5-devel` image and running

    ```
    nvidia-docker run --rm nvidia/cuda:7.5-devel nvidia-smi
    ```
3. To verify that the [cljs](https://github.com/graphistry/cljs) tests pass, run

    ```
    nvidia-docker run --rm graphistry/cljs:1.1 npm test
    ```

## 3 Set Up Graphistry

1. Create a directory for Graphistry products, like `/graphistry`, and unpack this release into a subdirectory. The top level directory can have config options like default passwords and default logging levels, and the release directory can have config overrides per release.
2. Go to this release's subdirectory, and run `load.sh` . This should load the airgapped Graphistry deploy into your local Docker engine.
3. Graphistry can terminate SSL, but by default will not. Follow the instructions in certs.txt for making certs, and for subsequent runtime parameters during launch, if you intend for Graphistry itself to terminate SSL.

## 4 Override Default Passwords

Launch the app with the environment variable `GRAPHISTRY_APP_CONFIG` including two unguessable strings for its `.API.CANARY` and its `.API.SECRET`, like so:

```
GRAPHISTRY_APP_CONFIG='{"API":{"CANARY":"123","SECRET":"456"}}' ./launch.sh
```

An API key, created for a user-identifying string (often in practice an email address), is that string, salted with a global salt, and then encrypted with a global password. The salt is `.API.CANARY`, and the password is `.API.SECRET`.

To avoid setting JSON configuration in the environment every time, write it to the file `httpd-config.json` in the parent directory of the release (so `echo '{"API":{"CANARY":"123","SECRET":"456"}}' > ../httpd-config.json`, for example).

## 5 Start The Graphistry Services
Once the containers are loaded and a new password has been written to config on disk, the app can be started (or restarted) with

```bash
./launch.sh
```

### Stop or Restart The Graphistry Services

Run `docker stop graphistry_httpd` or `docker restart graphistry_httpd`.

### Verify That Graphistry is Running

When you have launched the app, you should be able to point your browser to that machine via HTTP and receive a Getting Started page. Under the section "👁️ Learn More", click "See examples of Graphistry Visualizations", and click (for example) the _Protein Interactions_ at the lower right and ensure that you get a graph that resembles that image.

### Provision an API key for someone to use to upload data

Our upload service requires an API key for a user to upload a dataset.
These API keys are based on strings that should uniquely identify the user, often the user's email address.
To provision a new API key via command line, please run `./make-api-key.sh`, passing in a first command-line parameter of a base62-clean/url-encoded email address.

To provision an API key for 'someone@localhost', run

```
./make-api-key.sh someone@localhost
```

### Extra: Use Jupyter Notebooks With PyGraphsitry

These are public images, available on DockerHub, built on the Jupyter notebooks with pygraphistry installed, with several demo notebooks. Run

```
docker run -p 8888:8888  --restart=unless-stopped --name graphistry_notebook -d graphistry/jupyter-notebook:1.6
```

# Configuration and Maintenance

## View the Logs

Logs are mounted in the working directory of where you run launch.sh.

There are client logs, in clients/, server logs, in graphistry-json/, other assorted logs in central-app/ and worker/. Logs are in JSON. We recommend using [Bunyan](https://github.com/trentm/node-bunyan) to print logs in a more readable format. For instance, you can use

```
bunyan -o short < logfile
```

If you want debug logs, launch with the `GRAPHISTRY_APP_CONFIG` environment variable having a key in its JSON hash of `LOG_LEVEL` set to `DEBUG`; i.e.

```
GRAPHISTRY_APP_CONFIG='{"LOG_LEVEL":"DEBUG"}' ./launch.sh
```


## Workbooks Persistence

By default, the launch script will make a directory in its directory, `workbook_cache`, for all loaded workbooks.

If you make a new directory for each release that you deploy, and you are running an airgapped Graphistry, you may want to keep around the workbooks that you have previously made.

Please use the `GRAPHISTRY_WORKBOOK_CACHE` environment variable to set this to a directory of your choosing:

```
GRAPHISTRY_WORKBOOK_CACHE=/var/graphistry/workbooks GRAPHISTRY_APP_CONFIG='{"LOG_LEVEL":"INFO"}' ./launch.sh`
```

## Dataset Persistence

By default, the launch script will make a directory in its directory, `data_cache`, for all loaded datasets.

If you make a new directory for each release that you deploy, and you are running an airgapped Graphistry, you may want to keep around the datasets that you have previously loaded.

Please use the `GRAPHISTRY_DATA_CACHE` environment variable to set this to a directory of your choosing:

```
GRAPHISTRY_DATA_CACHE=/tmp/graphistry-data-cache ./launch.sh
```

## Read-Process-Discard for Datasets

Your datasets live in two places: in the Node process memory and in the `data_cache` directory. They are not written anywhere else.

If you would like to launch Graphistry so as to never write a dataset or workbook to disk, please first ensure that your machine has virtual memory turned off, to avoid swapping memory out to disk. (This is often a common requirement at an installation, and we defer to best practices therein.)

Please set the `GRAPHISTRY_DATA_CACHE` environment variable to some subdirectory of `/run/shm` (on Ubuntu 14.04 LTS) or `/dev/shm` (on RHEL 7.2).

This will ensure that Graphistry will never write your datasets to disk.

### Garbage Collection

For the discard part of read/process/discard, passing

```
GRAPHISTRY_APP_CONFIG='{"READ_PROCESS_DISCARD":true}'
```

to `launch.sh` causes datasets to be deleted immediately after the start of processing.
(If this is on, refreshing a page with a visualization will be an error, because that dataset will have been discarded.)

## Software Updates

New versions of Graphistry will be provided as an updated bundle containers. Each such bundle will contain a `load.sh` and `launch.sh` script to load the new containers in Docker and start the graphistry services, respectively.
