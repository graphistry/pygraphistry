'use strict';

var path = require('path');
var fs = require('fs');
var util = require('util');
var child_process = require('child_process');
var _ = require('lodash');
var Q = require('q');
var debug = require('debug')('graphistry:setup:tooling');

// The path passed to `ssh` to use for the Control Master socket
var defaultSshControlPath = '/tmp/deploy-ssh-github';
var wd = path.resolve(__dirname, '..', '..', '..');

// Exec a command and wrap its result in a promise
// From https://gist.github.com/Stuk/6226938
function exec(command, args, cwd, sshControlPath) {
    args = args || [];
    sshControlPath = sshControlPath || defaultSshControlPath;
    var deferred = Q.defer();

    var stderr_output = [];
    function combineStderr(data) { stderr_output.push(data); }

    if (!(_.isString(command) && _.isString(cwd))) {
        deferred.reject(new Error("Both command and working directory must be given, not " + command + " and " + cwd));
    } else if(!_.every(args, function(a){return _.isBoolean(a)||_.isString(a)||_.isNumber(a);})) {
        deferred.reject(new Error("All arguments must be a boolean, string or number"));
    } else {
        var command_with_args = [command].concat(args).join(' ');
        debug(util.format('(%s)$ %s', cwd, command_with_args));

        var proc_env = _.clone(process.env);
        proc_env['GIT_SSH_COMMAND'] = util.format(
            'ssh -o PermitLocalCommand=no -o ServerAliveInterval=0 -o ControlMaster=no -o ControlPath="%s"',
            sshControlPath);

        var proc = child_process.spawn(command, args, {
            cwd: cwd,
            env: proc_env,
            stdio: 'pipe'
        });
        // Speculatively create this so that its stack trace is useful (if we do it in .on('close'),
        // its stack trace will just be some async pipe close callback.)
        var spec_error = new Error('error executing command');

        proc.stdin.end();
        proc.stderr.on('data', combineStderr);

        proc.on("error", function (error) { deferred.reject(error); });
        proc.on("close", function(code) {
            proc.stderr.removeListener('data', combineStderr);
            if (code !== 0) {
                spec_error.message = util.format('command `%s` exited with error code %d (in %s)\n%s',
                    command_with_args, code, cwd, stderr_output.join('').replace(/^/mg, '\t| '));
                deferred.reject(spec_error);
            } else {
                deferred.resolve(proc);
            }
        });
    }

    return deferred.promise;
};

// [String] -> String -> PkgInfo
function getPkgInfo(roots, repo) {
    if (repo == 'ROOT') {
        return {
            name: 'ROOT',
            dependencies: _.map(roots, function (name) {
                return {name: name, repo: name, internal: true};
            })
        };
    }

    var pkgFile = path.resolve(wd, repo, 'package.json');
    var content = JSON.parse(fs.readFileSync(pkgFile, 'utf8'));

    // Consider devDependencies regular dependencies, since we install all packages in dev mode
    _.merge(content.dependencies, content.devDependencies);

    var deps = _.map(content.dependencies, function (url, name) {
        // Old Regex for explicit SSH link
        // var regex = /^git\+ssh:\/\/git@github.com:graphistry\/([\w-]+)\.git#master$/;

        // Regex for implicit github link, should also grab from old explicit link
        var regex = /graphistry\/([\w-]+)/;

        var result = url.match(regex);
        return {
            internal: result !== null,
            name: name,
            repo: (result !== null) ? result[1] : undefined,
            version: (result === null) ? url : undefined,
        };
    });

    return {
        name: content.name,
        dependencies: deps,
    };
}

// Clone a repo, fail if the directory already exists.
// Return promise containing the repo name when cloning has terminated
// String -> Promise[]
function clone(repo) {
    var repoURL = repo;
    if (repo.match(/^[-A-z0-9_]+$/)) {
        repoURL = 'git@github.com:graphistry/' + repo + '.git';
    } else {
        // Repo was the full path; extract the last path element:
        repo = repoURL.split('/').pop();
        // Remove suffix .git:
        repo = repo.split('.')[0];
    }
    var clone_path = path.resolve(wd, repo);
    var cmd = 'git';
    var clone_args = ['clone', repoURL];
    var pull_args = ['pull', '--ff-only', '--quiet'];

    // console.error('Cloning/updating repo "%s" (clone path: %s)', repo, clone_path);

    return Q.nfcall(fs.stat, clone_path)
        .then(
            function(/*stats*/) {
                console.error('Pulling repo "%s" (in %s)', repo, clone_path);

                return exec(cmd, pull_args, clone_path)
                    .catch(function(git_pull_err) {
                        var friendly_error = new Error(util.format(
                            'Error doing a `git pull --ff-only for repo "%s": %s',
                            repo, git_pull_err.message
                        ));

                        friendly_error.friendly_headline =
                            util.format('Repo "%s": could not do a fast-forward only pull', repo);
                        friendly_error.friendly_explanation =
                            ["You likely have uncommitted changes in that repo, or the merge with",
                             "the remote was more complicated than a fast-forward merge",
                             "(this tool will only allow fast-forward merges for safety reasons;",
                             "you should perform more complicated merges by hand.)"].join(' ');

                        throw friendly_error;
                    });
            },
            function(err) {
                console.error('Cloning repo "%s" (in %s)', repo, wd);
                return exec(cmd, clone_args, wd);
            }
        )
        .thenResolve(repo);
}


// Starts a `ssh` master connection to git@github.com. This will effectively pre-connect to GitHub,
// then allow subsequent connection (e.g. from `git {clone,pull}`) to tunnel over it without having
// to renegotiate a new SSH session each time, which may speed up cloning/pulling a bunch of repos.
function startSshControlMaster(socketPath) {
    socketPath = socketPath || defaultSshControlPath;

    return Q
        .nfcall(fs.stat, defaultSshControlPath)
        // socketPath exists; ControlMaster must already be running; don't start
        .then(function(stats) {
            return Q.value(true);
        })
        // socketPath does not exist; safe to start ControlMaster
        .catch(function(err) {
            // var out = fs.openSync('./out.log', 'a'), err = fs.openSync('./out.log', 'a');
            // console.log("--- Starting ssh master control connection to github.com");

            var proc = child_process.spawn('ssh',
                [
                    // Disable any local commands the user has configured to run after ssh connects,
                    // since this connection is going to be running headless in the background.
                    '-o', 'PermitLocalCommand=no',
                    // Don't send SSH protocol heartbeats (our connection is too short to merit it)
                    '-o', 'ServerAliveInterval=0',
                    // Make this a master connection (will connect and then background)
                    '-o', 'ControlMaster=yes',
                    // Keep the connection open for 30 seconds of idleness before closing
                    '-o', 'ControlPersist=30s',
                    '-o', util.format('ControlPath=%s', socketPath),
                    // Enable compression
                    '-o', 'Compression=yes',
                    '-o', 'CompressionLevel=6',
                    // Master connections don't need a TTY
                    '-o', 'RequestTTY=no',
                    '-N',   // do not execute a remote command
                    '-n',   // redirect STDIN from /dev/null (required for background SSH)
                    'git@github.com'
                ],
                {
                    detached: true,
                    stdio: ['ignore', 'ignore', 'ignore']
                }
            );

            // proc.on('error', function(err) {
            //     console.error("==== ssh master connection: execution of ssh command failed:", err);
            // });
            // proc.on('exit', function(code) {
            //     console.error(util.format("==== ssh master connection process exited with code %d", code));
            // });

            // Allow Node to exit even if the ssh master control process is still running
            proc.unref();

            // Wait 1 sec for socket to be set up
            return Q.delay(1000);
        });
}

function isInternal (x) { return x.internal; };

function buildDepTree(roots, repo, allExternals, done) {
    done = done || [];

    var info = getPkgInfo(roots, repo);

    var ideps = _.filter(info.dependencies, isInternal);
    var edeps = _.reject(info.dependencies, isInternal);
    var externals = _.map(edeps, function (d) {
        var ex = {name: d.name, version: d.version, source: repo};
        allExternals.push(ex);
        return ex;
    });
    var children = _.map(ideps, function (d) {
        if (_.contains(done, d.name)) {
            console.error('Cycle in dependencies', d, done);
            process.exit(1);
        }
        return buildDepTree(roots, d.repo, allExternals, done.concat([repo]));
    })

    return {
        name: info.name,
        repo: repo,
        deps: children,
        externals: externals,
    };
}

function topoSort(node) {
    function makeModule(node) {
        return {
            name: node.name,
            repo: node.repo,
            links: {
                internal: _.pluck(node.deps, 'name'),
                external: _.pluck(node.externals, 'name')
            }
        };
    }

    if (node.deps.length === 0) {
        return [makeModule(node)];
    }

    var sort = _.reduce(node.deps, function (sort, d) {
        var innerSort = topoSort(d);
        _.each(innerSort, function (i) { // Because _.union does not work with objects
            if (!_.find(sort, function (i2) { return i2.name === i.name }))
                sort.push(i);
        });
        return sort;
    }, []);
    sort.push(makeModule(node));
    return sort;
}

function isEquivalent(v1, v2) {
    var parts1 = v1.split('.');
    var parts2 = v2.split('.');

    var meat1 = parts1.slice(0, -1);
    var meat2 = parts2.slice(0, -1);

    return meat1.join() === meat2.join();
}

function distinctExternals(externals, checkVersionMismatch) {
    var mismatchs = {};
    var distinctExternals = _.reduce(externals, function (acc, dep) {
        var modifier = dep.version[0];
        if (modifier !== '~' && isNaN(modifier) && dep.version.indexOf('/') === -1) {
            console.warn('In %s, dependency %s is not set to bugfixes only (~): %s',
                         dep.source, dep.name, dep.version);
        }
        function sameDep(dep1, dep2) { return dep1.name === dep2.name; };
        var otherDep = _.find(acc, sameDep.bind('', dep));
        if (otherDep) {
            if (!isEquivalent(otherDep.version, dep.version)) {
                var entries = mismatchs[dep.name] || {};
                entries[dep.source] = dep.version;
                entries[otherDep.source] = otherDep.version;
                mismatchs[dep.name] = entries;
            }
            return acc;
        } else {
            return acc.concat([dep]);
        }
    }, []);

    if (checkVersionMismatch && Object.keys(mismatchs).length > 0) {
        console.warn('Version mismatch in dependencies:\n', mismatchs);
    }

    return distinctExternals;
}


function link(module, linkExternals) {
    if (module.name === 'ROOT') { return Q(); }

    console.log('Linking module "%s"...', module.name);

    var cmd = 'npm';
    var cwd = path.resolve(wd, module.repo);
    var toLink = module.links.internal;
    if (linkExternals) {
        toLink = toLink.concat(module.links.external);
    }

    //     `npm prune` module
    return exec(cmd, ['prune'], cwd)
        // `npm link <dependency>` for each unlinked dependency
        .thenResolve(
            _.chain(toLink)
            // Only `npm link <dependency>` if it's not already linked
            .filter(function filterUnlinkedDependencies(dependency) {
                try {
                    var depPath = path.resolve(cwd, 'node_modules', dependency);
                    // Already linked?
                    if(fs.lstatSync(depPath).isSymbolicLink()) {
                        return false;
                    } else {
                        console.log("  ...linking dependency \"%s\" (already installed, but not as a link)", dependency);
                        return true;
                    }
                } catch(ignore) {
                    console.log("  ...linking dependency \"%s\"", dependency);
                    return true;
                }
            })
            .map(function linkDependency(dependency) {
                return exec(cmd, ['link', dependency], cwd);
            })
            .value())
        .all()
        // `npm link` the module itself
        .then(function () {
            // Without `--no-bin-links`, npm will error if we ever `npm link` a module with binaries
            // twice, since it will refuse to overwrite the existing linked binary.
            return exec(cmd, ['link'], cwd);
        });
}

function installGlobally(externals) {
    var cmd = 'npm';

    return _.reduce(
        externals,
        function (wait, dep) {
            var args = ['install', '-g', dep.name + '@' + dep.version];
            return wait.then(function () { return exec(cmd, args, wd); });
        },
        Q());
}

module.exports = {
    getPkgInfo: getPkgInfo,
    clone: clone,
    startSshControlMaster: startSshControlMaster,
    buildDepTree: buildDepTree,
    topoSort: topoSort,
    distinctExternals: distinctExternals,
    link: link,
    installGlobally: installGlobally
};
