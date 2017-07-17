<!-- Generate PDF via 

1. MacDown

2. Or, 

docker run --rm -it -v $PWD:/source jagregory/pandoc -s hw_os_cloud_recommendations.md -o hw_os_cloud_recommendations.pdf 

-->



# Recommended Deployment Configurations: Client, Server Software, Server Hardware

In short:

* Client: Chrome/Firefox from the last 3 years, WebGL enabled, and 100KB/s download ability
* Server: x86 Linux server with 4+ CPU cores, 16+ GB CPU RAM (3GB per concurrent user), and 1+ Nvidia GPUs with 4+ GB RAM each (1+ GB per concurrent user)

## Client

The intuition is that a user can work with Graphistry if their environment supports Youtube, and even better, Netflix.

The Graphistry client runs in standard browser configurations:

* **Browser**: Chrome and Firefox from the last 3 years, and users regularly report success with other browsers like Safari.

* **WebGL**: WebGL 1.0 is required. This started shipping ~5 years ago, and most client devices, including phones and tablets, support it. Both integrated and discrete graphic cards work, and for any vendor supporting WebGL.

* **Network**: 100KB+/s download speeds, and we recommend 1MB/s if graphs with > 100K nodes and edges. 

* **Operating System**: All.

***Recommended***: Chrome from last 2 years and with a device from the last 4 years.


## Server Software: Cloud, OS, Docker, Avoiding Root Users

### Cloud

Graphistry runs on-premise and has been tested with Amazon EC2 and Microsoft Azure.

*Tested AWS Instances*:

* P2.xl ***Recommended***
* G2.2xl

*Tested Azure Instances*:

* NV6 ***Recommended***
* NC6

See the hardware provisioning section to pick the right configuration for you.

### OS & Docker

We regularly run on:

* Ubuntu Xenial 16.04 LTS
* RedHat RHEL 7.3 ***Recommended***

Both support nvidia-docker.

### User: Root vs. Not

Installing Docker, Nvidia drivers, and nvidia-docker currently all require root user permissions.

After installation, Graphistry can be installed and run as an unprivileged user, with access to nvidia-docker is installed, Graphistry can be installed and run as a regular user.

## Server: Hardware Capacity Planning

Graphistry utilization increases with the number of concurrent visualizations and the sizes of their datasets. 
Most teams will only have a few concurrent users and a few concurrent sessions per user. So, one primary server, and one spillover or dev server, gets a team far.

For teams doing single-purpose multi-year purchases, we generally recommend more GPUs and more memory: As Graphistry adds further scaling features, users will be able to upload more data and burst to more devices. 


### Network

A Graphistry server must support 1MB+/s per expected concurrent user. A moderately used team server may use a few hundred GB / month.

### GPUs & GPU RAM

The following Nvidia GPUs are known to work with Graphistry:

* Tesla: K40, K80, M40
* Pascal/DGX: P100 ***Recommended***

The GPU should provide 1+ GB of memory per concurrent user. For teams expecting to look at large datasets (1M-1B element graphs), we expect the consumable amount of memory per concurrent user to increase in 2018 by 100X, if desired.

### CPU Cores & CPU RAM

CPU cores & CPU RAM should be provisioned in proportion to the number of GPUs and users:

* CPU Cores: We recommend 4-6 x86 CPU cores per GPU
* CPU RAM: We recommend 6 GB base memory and at least 16 GB total memory for a single GPU system. For balanced scaling, 3 GB per concurrent user or 3X the GPU RAM.

### CPU-Only

For development purposes such as testing, a CPU-only mode (for machines without a GPU) is available.