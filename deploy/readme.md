## Setting up Local Development

### Preliminaries

Install the following (apt-get/brew) packages:
 - `git`
 - `node` (0.10.x) and `node-legacy` (if on Ubuntu)
 - `glfw` (3.x), `anttweakbar`, and `freeimage` (on Mac)

Then install the following NPM packages globally (`npm install -g XXX`):
 - `grunt` and `grunt-cli`
 - `browserify`
 - `less`

### The Graphistry Stack

1. Create a new empty directory (WD/ in this example).

2. Clone this repository inside WD: `git clone git@github.com:graphistry/deploy.git`. You know have WD/deploy.

3. Run NPM install the setup script: `cd deploy/tools/setup && npm install`.

4. Run setup.js from `WD/deploy` to clone and link all remaining repositories: `cd deploy && ./tools/setup/setup.js --clone --link`. You can add the `--shared` flag to install all external (non-graphistry) dependencies globally, thus avoiding having multiple copies of the same libraries, one in each repository.

5. Run `./tools/check.sh` (still in deploy) to run all tests.

## Login ##

`ssh -A leo@54.183.193.90 -p 61630`

## To set up a fresh machine:##

1. Create new machine in EC2, provision with ansible_id_rsa key.

2. Log onto the box by hand:

`ssh -A -i ansible_id_rsa.pem ubuntu@XXXXXXX`

`sudo apt-get update && sudo apt-get install linux-headers-generic`

3. Exit box, go to deploy repo

4. Comment out ansible port in hosts.yml (box is started with port 22 open instead of 61630)

5. `ansible-playbook -i hosts system.yml -vvvv --tags ssh`

6. Uncomment ansible port.

7. `ansible-playbook -i hosts system.yml -vvvv --tags YOUR_BOX'S_TAG`

8. It'll reboot. Then deploy:

`ansible-playbook -i hosts system.yml -vvvv --tags YOUR_BOX'S_TAG --skip-tags provision`

9. You're done.

To start the workers/servers:
```
sudo supervisorctl start all
sudo supervisorctl stop all
sudo supervisorctl restart all
```

Logs:

`tail -f /var/log/worker/worker-10000.log`

## To Deploy:

### Staging:

1. Commit your changes locally to the master branch of each repository, **test**, and push.

2. Pull latest `deploy` repo

3. run `stage-deploy.sh` to deploy to the staging server

4. Verify your changes at the url pointed to in the slack #ansible channel

### Prod:

Regular version:

`./prod-deploy.sh`

Extended version (if you know what you're doing):

`ansible-playbook system.yml -vv --skip-tags provision,staging-slack -i hosts -l prod`

Verify your changes at the url pointed to in the slack #ansible channel


### Live Editing

On the servers with the new deploy workflow, we now clone the graphistry apps into `/var/graphistry/`. Feel free to do a `git pull` on any of these, and/or edit the files by hand and push them back to GitHub. Be sure to do a `sudo supervisorctl restart viz-worker:*` (for the workers, or `sudo supervisorctl restart central` for central.)



##Localdev (defunct for now):

```
vagrant up dev
vagrant ssh
sudo apt-get install linux-headers-generic
sudo ansible-playbook -i hosts system.yml -vvvv --tags localdev --skip-tags splunk,ssh
reboot
sudo ansible-playbook -i hosts system.yml -vvvv --tags localdev --skip-tags splunk,ssh,worker-reboot
vagrant ssh
```

## To keep in mind:
- if something isn't right on staging, and you cannot replicate the error locally, it's okay to try and fix it on the live EC2 staging server. However, please don't get in the habit of developing exclusively on the staging server.
- join and inspect the #ansible slack channel, where all deployment notifications are automatically posted. Ansible posts in slack every time a staging / prod deploy kicks off, so we all know who is deploying where at any given time.
- if someone is currently in the process of deploying, do not deploy until they have finished. You can monitor the process of deploys in #ansible on Slack - Ansible posts to Slack every time a deploy begins and finishes, so there should be good situational awareness here.
- if you've got a commit on master, please make sure it's deployed and verified to work on prod as opposed to leaving it there to be deployed later
- CI and testing is on the way!

Happy deploying!
