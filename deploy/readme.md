# Graphistry Deploy

This repo contains the Docker files and Ansible scripts which deploy our application stack to AWS. It also contain the `setup.js` tool, which can download/update and link our application stack on both local and remote systems.

Additionally, it contains documentation pertaining to our overall application stack, including installing and running it. Documentation for individual applications should be kept in the repositories for those applications.

## Documentation

* [Local Development](https://github.com/graphistry/deploy/blob/master/documentation/local-dev.md): How to download and install the Graphistry application stack so that you can do local development.
* [AWS](https://github.com/graphistry/deploy/blob/master/documentation/aws.md): How to deploy the Graphistry stack to Amazon Web Services for both staging and production.
* [Troubleshooting](https://github.com/graphistry/deploy/blob/master/documentation/troubleshooting.md): What to do if the server stops working.
* [Manual Testing](https://docs.google.com/document/d/1J7UgXXXs5LujC6Nl6st0ooavfzRAZKI1ZSzl6n_Z90c/edit#heading=h.z8ufwwipb9wm): Demos and tasks to try before prod deploys.
* [Jenkins!](http://deploy.graphistry.com/): CI server for testing history and remote builds

## User Documentation

When deployed as a container, admins get the following docs:

* [Instructions.md](https://github.com/graphistry/deploy/blob/master/dockerfiles/instructions.md)

## Modifying this repository

Commits to this repository are generally infrequent; as such, our Jenkins tooling uses the `master` branch of this repository to deploy the app, customarily one tests changes immediately after they land, and this has worked well thus far.
All commits worth discussing should be mentioned in #infrastructure or proposed as a PR.
