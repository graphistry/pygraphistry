# Local Development

## :warning: ANSIBLE WARNING! (2/4/16) :warning:

Currently, our Ansible playbooks are *only* compatible with **Ansible 1.9.x**. Other version (namely 1.8 and 2.0) are currently *incompatible*, and you will not be unable to run a deploy using them.

**Mac OS X**: Unfortunately, the Homebrew formula `ansible` has already been changed to v2.0.0, so *upgrades will leave you with the wrong version!* Run the following commands to install the proper version:

    brew uninstall ansible; brew tap homebrew/versions; brew install homebrew/versions/ansible19

 **Ubuntu**: Since the official Trusty apt repo is too old, and the official Ansible PPA repo is too new, you need to install Ansible via `pip`. Uninstall the ansible apt package if you already have it, and then run this command to install the proper version:

    sudo apt-get install python-pip; sudo pip install ansible==1.9.4 jinja2


## Install

### Mac

1. Install [Xcode from the App Store](https://itunes.apple.com/us/app/xcode/id497799835?mt=12), and **launch it at least once** to agree to its SDK, and let it auto-install the command line build tools.
2. Install [Homebrew](https://github.com/Homebrew/homebrew):`/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`
3. Install required software via Homebrew: `brew tap homebrew/versions; brew install n git homebrew/versions/ansible19 pigz`
4. Use `n` to install Node.js. First run `n ls` to find the latest `0.10.x` release, then run this (replace `<version>` with the correct version number): `n <version>`
5. Install a compatible version of `npm` (2.x.x): `npm install -g npm@latest-2`
6. Make sure your computer's SSH public key is [authorized on your GitHub account](https://github.com/settings/ssh). Instructions for doing so can be found [here](https://help.github.com/articles/generating-an-ssh-key/).
7. Create a new directory somewhere to download all of our Graphistry code to, and `cd` into it.
8. Clone the deploy repo (aka this repo): `git clone git@github.com:graphistry/deploy.git`
9. Use our handy script to clone and install all of our important repos: `./deploy/tools/setup/setup.js --clone --link`
10. You can deploy the latest code to the servers by running `./deploy/stage-deploy.sh` and `./deploy/prod-deploy.sh`, to deploy to staging or production, respectively. *Note: the HEAD of the `master` branch of each repo is what is deployed; it is not currently possible to deploy other branches/commits.*


### Ubuntu

*Warning: Ubuntu documentation is a work in progress, and is currently incomplete.*

It is reccomended to use the [NodeSource Node.js distribution](https://github.com/nodesource/distributions) for Node.js rather than the default Ubuntu one. This is what the EC2 servers use, so it will most closely match our production environment.

```bash
sudo apt-get install python-pip; sudo pip install ansible==1.9.4 jinja2
curl -sL https://deb.nodesource.com/setup | sudo bash -
sudo apt-get install -y nodejs
```


## Server Login

We run all of our SSH daemons on port 61630. You should have a user account with your SSH key installed already. If not, talk to Matt. Also make sure to turn on SSH authentication forwarding when you login (`-A`) to make GitHub cloning work.

For example, your SSH command to connect to staging may be `ssh -A -p 61630 mtorok@staging.graphistry.com`.
