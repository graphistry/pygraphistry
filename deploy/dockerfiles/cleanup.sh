#!/bin/sh -xe
docker rmi `docker images | awk '{print $3}'` || true
