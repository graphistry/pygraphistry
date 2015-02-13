'use strict';

var path = require('path');
var fs = require('fs');
var child_process = require('child_process');
var _ = require('underscore');
var Q = require('q');

var wd = path.resolve(__dirname, '..', '..', '..');
var DEBUG = false;

var errorHandler = function (err) {
    console.error('Error', err, (err||{}).stack);
};

// Exec a command and wrap its result in a promise
// From https://gist.github.com/Stuk/6226938
function exec(command, args, cwd) {
    if (!command || !cwd) {
        return Q.reject(new Error("Both command and working directory must be given, not " + command + " and " + cwd));
    }
    if (args && !args.every(function (arg) {
        var type = typeof arg;
        return type === "boolean" || type === "string" || type === "number";
    })) {
        return Q.reject(new Error("All arguments must be a boolean, string or number"));
    }

    var deferred = Q.defer();

    console.log("+", command, args.join(" "), "# in", cwd);

    var proc = child_process.spawn(command, args, {
        cwd: cwd,
        stdio: DEBUG ? "inherit" : "ignore"
    });
    proc.on("error", function (error) {
        deferred.reject(new Error(command + " " + args.join(" ") + " in " + cwd + " encountered error " + error.message));
    });
    proc.on("exit", function(code) {
        if (code !== 0) {
            deferred.reject(new Error(command + " " + args.join(" ") + " in " + cwd + " exited with code " + code));
        } else {
            deferred.resolve();
        }
    });
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
    var deps = _.map(content.dependencies, function (url, name) {
        var regex = /^git\+ssh:\/\/git@github.com:graphistry\/([\w-]+)\.git#master$/;
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
// Return promise contaning the repo name when cloning has terminated
// String -> Promise[]
function clone(repo) {
    var cmd = 'git';
    var args = ['clone', 'git@github.com:graphistry/' + repo + '.git'];
    return exec(cmd, args, wd).then(function () {
        return repo;
    }).fail(errorHandler);
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
        })
        return sort;
    }, []);
    sort.push(makeModule(node));
    return sort;
}


function distinctExternals(externals, checkVersionMismatch) {
    var mismatchs = {};
    var distinctExternals = _.reduce(externals, function (acc, dep) {
        function sameDep(dep1, dep2) { return dep1.name === dep2.name; };
        var otherDep = _.find(acc, sameDep.bind('', dep));
        if (otherDep) {
            if (otherDep.version !== dep.version) {
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
    if (module.name === 'ROOT')
        return Q();

    var cmd = 'npm';
    var cwd = path.resolve(wd, module.repo);

    var toLink = module.links.internal;
    if (linkExternals) {
        toLink = toLink.concat(module.links.external);
    }

    return Q.all(
        _.map(toLink, function (target) {
            var args = ['link', target];
            return exec(cmd, args, cwd).fail(errorHandler);
        })
    ).then(function () {
        var args = ['link'];
        return exec(cmd, args, cwd).fail(errorHandler);
    }).fail(errorHandler);
}

function installGlobally(externals) {
    var cmd = 'npm';

    return _.reduce(externals, function (wait, dep) {
        var args = ['install', '-g', dep.name + '@' + dep.version];
        return wait.then(function () {
            return exec(cmd, args, wd).fail(errorHandler);
        });
    }, Q()).fail(errorHandler);
}

module.exports = {
    getPkgInfo: getPkgInfo,
    clone: clone,
    buildDepTree: buildDepTree,
    topoSort: topoSort,
    distinctExternals: distinctExternals,
    link: link,
    installGlobally: installGlobally
}
