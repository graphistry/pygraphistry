#!/bin/sh -x -e
docker rmi `docker images | awk '{print $3}'`
