# Local Development

## Setup

### Mac

Mac sure you've installed [Homebrew](https://github.com/Homebrew/homebrew), then install our dependencies with the below commands.

**Note**: make sure you *don't* install the `node` package (or `brew uninstall node` if you have it installed.) Homebrew now only includes Node.js v0.12.x, and some of our modules only work with v0.10.x. We will use `n` to install Node.js instead.

```bash
brew install git freeimage glfw3 anttweakbar ansible
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
