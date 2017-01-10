#!/usr/bin/env node
'use strict';

var util = require('util');
var _ = require('lodash');
var Q = require('q');
var tooling = require('./tooling.js');
var debug = require('debug')('graphistry:setup:main');

var argv = require('yargs')
    .usage('Download and link the Graphistry stack')
    .example('$0 --pull --link', 'Setup a local dev environment from scratch')

    .default('pull', false)
    .alias('pull', 'p')
    .alias('pull', 'clone')
    .alias('pull', 'c')
    .describe('pull', 'Pulls all Graphistry repos, cloning any that are missing locally')

    .default('link', false)
    .alias('link', 'l')
    .describe('link', 'Link/install local checkout of Graphistry\'s repo')

    .default('ignore-errors', false)
    .alias('ignore-errors', 'i')
    .describe('ignore-errors', 'Ignore pull/clone errors, rather than exiting')

    .default('dry-run', false)
    .alias('dry-run', 'd')
    .describe('dry-run', 'Do not execute link/install commands, print installation dependencies instead.')

    .default('shared', false)
    .alias('shared', 's')
    .describe('shared', 'install external dependencies globally, enabling cross repos sharing')

    .default('versions', false)
    .alias('versions', 'v')
    .describe('versions', 'Report libraries imported using different/mismatched versions')

    .boolean(['pull', 'link', 'ignore-errors', 'dry-run', 'shared', 'versions'])
    .help('help')
    .alias('help', 'h')
    .argv;

var roots = ['central', 'viz-app', 'pivot-app'];

var errorHandler = function (err) {
    if (err.friendly_headline) {
        console.error('****** Error: %s', err.friendly_headline);
        if(err.friendly_explanation) { console.error(err.friendly_explanation); }
    } else {
        console.error(err.stack);
    }
};


function cloneOne(repo) {
    return tooling.clone(repo)
        .then(
            _.identity,
            function handleCloneRepoError(err) {
                if(argv['ignore-errors']) {
                    // Print error info, but just return the repo name as normal
                    errorHandler(err);
                    console.error('------ Ignoring error because --ignore-errors used');
                    return repo;
                } else {
                    // Re-throw the error so that it halts execution
                    throw err;
                }
            });
}

// Recursively clone a repository and its (run-time) dependencies.
// [String] -> Promise[]
function cloneAll(stack, done) {
    if (stack.length === 0) { return Q(); }

    var covered = (done || []).concat(stack);

    return tooling.startSshControlMaster()
        .then(function() {
            var cloneReposPromises = _.map(stack, function(repo) { return cloneOne(repo); });
            return Q.all(cloneReposPromises);
        })
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
        .then(function cloneRecurse(todo) {
            return cloneAll(todo, covered);
        });
}

// Link all repositories following the topological order given
// [Repos] -> Boolean -> Promise[]
function linkAll(repos, installExternalGlobally) {
    // If `-c` was also given, then we've already printed `cloneAll()` messages, so write a `\n`
    if (argv.pull) { console.log(''); }

    var allExternals = [];
    var depTree = tooling.buildDepTree(roots, 'ROOT', allExternals);
    debug('Dependencies tree', JSON.stringify(depTree, null, 2));

    var sort = tooling.topoSort(depTree);
    debug('Sort', sort);
    if (argv['dry-run']) {
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


if (argv.versions) {
    var allExternals = [];
    var depTree = tooling.buildDepTree(roots, 'ROOT', allExternals);
    tooling.distinctExternals(allExternals, true);
    return;
}


(argv.pull ? cloneAll(roots) : Q())
    .then(function() {
        return argv.link ? linkAll(roots, argv.shared) : Q();
    })
    .done(_.noop, function(err) {
        errorHandler(err);
        process.exit(1);
    });
