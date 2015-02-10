#!/usr/bin/env node
'use strict';

var path = require('path');
var _ = require('underscore');
var Q = require('q');
var tooling = require('./tooling.js');
var debug = require('debug')('graphistry:setup');

var argv = require('yargs')
  .usage('Download and link the Graphistry stack')
  .example('$0 -c -l', 'Setup a local dev environment from scratch')
  .default('c', false)
  .alias('c', 'clone')
  .describe('c', 'Clone all of Graphistry\'s repos')
  .default('l', false)
  .alias('l', 'link')
  .describe('l', 'Link/install local checkout of Graphistry\'s repo')
  .default('s', false)
  .alias('s', 'shared')
  .describe('s', 'install external dependencies globally, enambing cross repos sharing')
  .default('v', false)
  .alias('v', 'version')
  .describe('v', 'Report libraries imported using different/mismatched versions')
  .boolean(['c', 'l', 's', 'v'])
  .help('help')
  .argv;

var roots = ['central', 'viz-server'];

var errorHandler = function (err) {
    console.error('Error', err, (err||{}).stack);
};

// Recursively clone a repository and its (run-time) dependencies.
// [String] -> Promise[]
function cloneAll(stack, done) {
    if (stack.length === 0)
        return Q();

    var newRepos = _.chain(stack).map(function (repo) {
        return tooling.clone(repo);
    }).map(function (proc) {
        return proc.then(function (repo) {
            var deps = tooling.getPkgInfo(roots, repo).dependencies;
            return  _.chain(deps).where({internal: true}).pluck('repo').value();
        });
    }).value();

    var covered = (done || []).concat(stack);
    return Q.all(newRepos).then(function (reposArray) {
        var todo = _.chain(reposArray).flatten().uniq().difference(covered).value();
        return cloneAll(todo, covered);
    }).fail(errorHandler);
}

// Link all repositories following the topological order given
// [Repos] -> Boolean -> Promise[]
function linkAll(repos, installExternalGlobally) {
    var allExternals= [];
    var depTree = tooling.buildDepTree(roots, 'ROOT', allExternals);
    debug('Dependencies tree', JSON.stringify(depTree, null, 2))

    var sort = tooling.topoSort(depTree);
    debug('Sort', sort);

    var distinctExternals = tooling.distinctExternals(allExternals, false);

    return Q().then(function () {
        if (installExternalGlobally) {
            return tooling.installGlobally(distinctExternals);
        }
    }).then(function () {
        return _.reduce(sort, function (prev, module) {
            return prev.then(function () {
                return tooling.link(module, installExternalGlobally);
            });
        }, Q()).fail(errorHandler);
    });
}

if (argv.v) {
    var allExternals= [];
    var depTree = tooling.buildDepTree(roots, 'ROOT', allExternals);
    tooling.distinctExternals(allExternals, true);
    return;
}

Q().then(function () {
    if (argv.c) {
        return cloneAll(roots);
    }
}).then(function () {
    if (argv.l) {
        return linkAll(roots, argv.s)
    }
}).fail(errorHandler);



