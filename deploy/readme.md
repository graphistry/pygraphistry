# Local Development

## Setup

### Mac

Mac sure you've installed [Homebrew](https://github.com/Homebrew/homebrew), then install our dependencies with the below commands. 

**Note**: make sure you *don't* install the `node` package (or `brew uninstall node` if you have it installed.) Homebrew now only includes Node.js v0.12.x, and some of our modules only work with v0.10.x. We will use `n` to install Node.js instead. 

```bash
brew install git freeimage glfw anttweakbar n ansible
# Use this command to find the latest 0.10.x version of Node
n ls
# Assuming "0.10.36" is the latest 0.10.x version found above...
n 0.10.36
```


### Ubuntu

It is reccomended to use the [NodeSource Node.js distribution](https://github.com/nodesource/distributions) for Node.js rather than the default Ubuntu one. This is what the EC2 servers use, so it will most closely mathc our production environment.

```bash
sudo apt-get install git ansible
curl -sL https://deb.nodesource.com/setup | sudo bash -
sudo apt-get install -y nodejs
```

### Common

Install the `grunt` command:

```bash
npm install --global grunt-cli
```

Life will also be easier if you add the local packages npm binaries to your path. However, these change based on your current directory. I accomplish this by adding the following to my `.bash_profile`:

```bash
_npm_bin_path() {
    # Remove old npm bin paths from $PATH
    local npm_PATH="$(printf '%s' "$PATH" | sed -E -e 's|(:)?[^:]*/node_modules/\.bin||g')"
    # Ask npm for the local bin path. npm just guesses though, so we also verify it exists before adding it to $PATH
    [[ -d "$(npm bin)" ]] && npm_PATH="$npm_PATH:$(npm bin)"
    export PATH="$npm_PATH"
}
export PROMPT_COMMAND="${PROMPT_COMMAND:-:} ; _npm_bin_path"
```


## The Graphistry Stack

1. Create a new empty directory (WD/ in this example).
2. Clone this repository inside WD: `git clone git@github.com:graphistry/deploy.git`. You know have WD/deploy.
3. Run NPM install the setup script: `cd deploy/tools/setup && npm install`.
4. Run setup.js from `WD/deploy` to clone and link all remaining repositories: `cd deploy && ./tools/setup/setup.js --clone --link`. You can add the `--shared` flag to install all external (non-graphistry) dependencies globally, thus avoiding having multiple copies of the same libraries, one in each repository.
5. Run `./tools/check.sh` (still in deploy) to run all tests.


## EC2 Server Login

We run all of our SSH deamons on port 61630. You should have a user account with your SSH key installed already. If not, talk to Matt. Also make sure to turn on SSH authentication forwarding when you login (`-A`) to make GitHub cloning work.

For example, your SSH command to connect to staging may be `ssh -A -p 61630 mtorok@staging.graphistry.com`.


# EC2

## Provisioning

1. Create new EC2 instance with the following settings:
 * Region: us-west-1 (N. California)
 * AMI: Ubuntu 14.04 LTR "Trusty" with HVM as the AMI
 * Instance type: g2.2xlarge
 * Network: vpc-8045a9ef
 * Subnet: subnet-b71428f1
 * Placement group: graphistry-dev
 * IAM role: either 'central-server', 'viz-worker' or 'central-worker-combined-server', depending on what this instance will run
 * Storage: 20 GB SSD
 * Name: cluster-name/hostname (e.g., "production/labs")
 * Security groups: "Select an existing security group" and select sure default, default-ssh, Unique_SSH, central-server and/or worker (depending on the role of the machine.)
 * After you hit "Launch", for SSH key pair, select "Choose an existing key pair", and the "ansible\_id\_rsa" key
2. After launch, assign the inastance an Elastic IP (provision a new one if needed, or re-assign an existing one if replacing an existing instance.)
3. Note both the public (Elastic) IP and the private IP in the ec2 console. 
4. If this instance should be accessible at some new domain name, go to the AWS Route53 console, then:
  * Select the "graphistry.com." *public* hosted zone and go to its record set
  * Create a record set
  * Name: the hostname of the machine (e.g., if you want this machine to be accessible as worker09.graphistry.com, name the record "worker09")
  * Type: A - IPv4
  * Alias: no
  * TTL: 300 (i.e., 5 minutes)
  * Value: enter the public (Elastic) IP of the machine
  * Routing policy: simple
  * Click "Create" and then run `dig worker09.graphistry.com` (or whatever the address of the machine should be) on your local machine. Within 5 minutes, you should see the IP associated with the domain.
5. Repeat the above steps, but select the graphistry.com *private* hosted zone, and enter the instance's private IP as the record value. Log into another ec2 instance in our VPC and run `dig worker09.graphistry.com`. Within 5 minutes, you should see the *private* IP returned for that DNS record.
6. If this instance should be accessible at some existing domain name instead, go to Route53 and change the "value" for that record in both public and private hosted zones to be the public (Elastic) and private IPs of the new instance, respectively.
7. If this instance should join an existing cluster (production, staging), open the `./production` or `./staging` (or whatever) inventory file, and add this instance's domain name to the `[central]` and/or `[workers]` section.
8. If this instance should be part of a new cluster, create a new cluster inventory file at the top level of this repo. Copy an existing one as a template (but be sure to remove the other cluster's machines from the new inventory file.) Make sure to set the `cluster`, `mongodb_url`, and `mongodb_dbname` variables in the inventory file.
9. Run Ansible in full deploy mode. Assuming you're in this repo's directory, run `ansible-playbook site.yml -i <inventory file>`, where `<inventory file>` is the filename to your intended cluster's inventory (`production`, `staging`, etc.)
10. Ansible should run straight-through without errors. Your new machine should be up and working at this point.


## Deploying

1. Commit and push your git changes for all your Graphistry repos first.
2. Run `./staging-deploy.sh` or `./prod-deploy.sh` from this repo's directory

*Not Reccomended:* If you want to bypass the `check.sh` script to deploy with outstanding commits, you can run Ansible directly: `ansible-playbook site.yml -i <inventory file> --tags deploy`.

If you're creating a new cluster, copy the `staging-deploy.sh` script, name it after your cluster, and replace the `ansible-playbook` command within to use your cluster's inventory file.


## Controlling processes

To start the workers/servers:
```
sudo supervisorctl start all
sudo supervisorctl stop all
sudo supervisorctl restart all
```

Logs:

`tail -f /var/log/worker/worker-10000.log`


### Live Editing

On the servers with the new deploy workflow, we now clone the graphistry apps into `/var/graphistry/`. Feel free to do a `git pull` on any of these, and/or edit the files by hand and push them back to GitHub. Be sure to do a `sudo supervisorctl restart viz-worker:*` (for the workers, or `sudo supervisorctl restart central` for central.)


## To keep in mind:
- if something isn't right on staging, and you cannot replicate the error locally, it's okay to try and fix it on the live EC2 staging server. However, please don't get in the habit of developing exclusively on the staging server.
- join and inspect the #ansible slack channel, where all deployment notifications are automatically posted. Ansible posts in slack every time a staging / prod deploy kicks off, so we all know who is deploying where at any given time.
- if someone is currently in the process of deploying, do not deploy until they have finished. You can monitor the process of deploys in #ansible on Slack - Ansible posts to Slack every time a deploy begins and finishes, so there should be good situational awareness here.
- if you've got a commit on master, please make sure it's deployed and verified to work on prod as opposed to leaving it there to be deployed later
- CI and testing is on the way!

Happy deploying!
