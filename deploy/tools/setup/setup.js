#!/usr/bin/env node
'use strict';

var util = require('util');
var _ = require('lodash');
var Q = require('q');
var tooling = require('./tooling.js');
var debug = require('debug')('graphistry:setup:main');

var argv = require('yargs')
    .usage('Download and link the Graphistry stack')
    .example('$0 -c -l', 'Setup a local dev environment from scratch')
    .default('c', false)
    .alias('c', 'clone')
    .describe('c', 'Clone all of Graphistry\'s repos')
    .default('l', false)
    .alias('l', 'link')
    .describe('l', 'Link/install local checkout of Graphistry\'s repo')
    .default('d', false)
    .alias('d', 'dry-run')
    .describe('d', 'Do not execute link/install commands, print installation dependencies instead.')
    .default('s', false)
    .alias('s', 'shared')
    .describe('s', 'install external dependencies globally, enambing cross repos sharing')
    .default('v', false)
    .alias('v', 'versions')
    .describe('v', 'Report libraries imported using different/mismatched versions')
    .boolean(['c', 'l', 's', 'v'])
    .help('help')
    .argv;

var roots = ['central', 'viz-server'];

var errorHandler = function (err) {
    if (err.friendly_headline) {
        console.error(util.format('\n****** Error\n%s', err.friendly_headline));
        if (err.friendly_explanation) {
            console.error(util.format('\n%s', err.friendly_explanation));
        }
    } else {
        console.error(err.stack);
    }
};

// Recursively clone a repository and its (run-time) dependencies.
// [String] -> Promise[]
function cloneAll(stack, done) {
    if (stack.length === 0) { return Q(); }

    var covered = (done || []).concat(stack);

    return Q
        .all(_.map(stack, function (repo) { return tooling.clone(repo); }))
        .then(function getClonedDeps(repos) {
            return _.chain(repos)
                .map(function (clonedRepo) { return tooling.getPkgInfo(roots, clonedRepo); })
                .map('dependencies')
                .flatten()
                .where({internal: true})
                .pluck('repo')
                .uniq()
                .difference(done)
                .value();
        })
        // .fail(errorHandler);
        .then(function cloneRecurse(todo) {
            return cloneAll(todo, covered);
        });
}

// Link all repositories following the topological order given
// [Repos] -> Boolean -> Promise[]
function linkAll(repos, installExternalGlobally) {
    // If `-c` was also given, then we've already printed `cloneAll()` messages, so write a `\n`
    if (argv.c) { console.log(''); }

    var allExternals = [];
    var depTree = tooling.buildDepTree(roots, 'ROOT', allExternals);
    debug('Dependencies tree', JSON.stringify(depTree, null, 2));

    var sort = tooling.topoSort(depTree);
    debug('Sort', sort);
    if (argv.d) {
        console.log(JSON.stringify(sort, null, 2));
        return;
    }

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
        }, Q()); // .fail(errorHandler);
    });
}

if (argv.v) {
    var allExternals = [];
    var depTree = tooling.buildDepTree(roots, 'ROOT', allExternals);
    tooling.distinctExternals(allExternals, true);
    return;
}

Q().then(function () {
        if (argv.c) {
            return tooling.startSshControlMaster()
                .then(function () {
                    return cloneAll(roots);
                });
        }
    })
    .done(
        function () {
            if (argv.l) { return linkAll(roots, argv.s); }
        },
        errorHandler
    );
