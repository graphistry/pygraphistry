#!/bin/sh -ex
for lxc in *.lxc.gz ; do docker load -i $lxc ; done
