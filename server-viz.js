#!/usr/bin/env node
'use strict';

//Set jshint to ignore `predef:'io'` in .jshintrc so we can manually define io here
/* global -io */

var Rx          = require('rx');
var _           = require('underscore');
var Q           = require('q');
var fs          = require('fs');
var path        = require('path');
var extend      = require('node.extend');
var rConf       = require('./js/renderer.config.js');
var lConf       = require('./js/layout.config.js');
var loader      = require('./js/data-loader.js');
var driver      = require('./js/node-driver.js');
var persist     = require('./js/persist.js');
var workbook    = require('./js/workbook.js');
var labeler     = require('./js/labeler.js');
var encodings   = require('./js/encodings.js');
var DataframeMask = require('./js/DataframeMask.js');
var TransactionalIdentifier = require('./js/TransactionalIdentifier');
var vgwriter    = require('./js/libs/VGraphWriter.js');
var compress    = require('node-pigz');
var config      = require('config')();
var util        = require('./js/util.js');
var ExpressionCodeGenerator = require('./js/expressionCodeGenerator');

var log         = require('common/logger.js');
var logger      = log.createLogger('graph-viz:driver:viz-server');
var perf        = require('common/perfStats.js').createPerfMonitor();

/**** GLOBALS ****************************************************/


var saveAtEachStep = false;
var defaultSnapshotName = 'snapshot';


/**** END GLOBALS ****************************************************/


/** Given an Object with buffers as values, returns the sum size in megabytes of all buffers */
function sizeInMBOfVBOs(VBOs) {
    var vboSizeBytes =
        _.reduce(
            _.pluck(_.values(VBOs.buffers), 'byteLength'),
            function(acc, v) { return acc + v; }, 0);
    return (vboSizeBytes / (1024 * 1024)).toFixed(1);
}

// TODO: Dataframe doesn't currently support sorted/filtered views, so we just do
// a shitty job and manage it directly out here, which is slow + error prone.
// We need to extend dataframe to allow us to have views.
function sliceSelection(dataFrame, type, indices, start, end, sort_by, ascending, searchFilter) {

    if (searchFilter) {
        searchFilter = searchFilter.toLowerCase();
        var newIndices = [];
        _.each(indices, function (idx) {
            var row = dataFrame.getRowAt(idx, type);
            var keep = false;
            _.each(row, function (val/*, key*/) {
                if (String(val).toLowerCase().indexOf(searchFilter) > -1) {
                    keep = true;
                }
            });
            if (keep) {
                newIndices.push(idx);
            }
        });
        indices = newIndices;
    }

    var count = indices.length;

    if (sort_by === undefined) {
        return {count: count, values: dataFrame.getRows(indices.slice(start, end), type)};
    }

    // TODO: Speed this up / cache sorting. Actually, put this into dataframe itself.
    // Only using permutation out here because this should be pushed into dataframe.
    var sortCol = dataFrame.getColumnValues(sort_by, type);
    var sortToUnsortedIdx = dataFrame.getHostBuffer('forwardsEdges').edgePermutationInverseTyped;
    var taggedSortCol = _.map(indices, function (idx) {
        if (type === 'edge') {
            return [sortCol[sortToUnsortedIdx[idx]], idx];
        } else {
            return [sortCol[idx], idx];
        }

    });

    var sortedTags = taggedSortCol.sort(function (val1, val2) {
        var a = val1[0];
        var b = val2[0];
        if (typeof a === 'string' && typeof b === 'string') {
            return (ascending ? a.localeCompare(b) : b.localeCompare(a));
        } else if (isNaN(a) || a < b) {
            return ascending ? -1 : 1;
        } else if (isNaN(b) || a > b) {
            return ascending ? 1 : -1;
        } else {
            return 0;
        }
    });

    var slicedTags = sortedTags.slice(start, end);
    var slicedIndices = _.map(slicedTags, function (val) {
        return val[1];
    });

    return {count: count, values: dataFrame.getRows(slicedIndices, type)};
}

VizServer.prototype.resetState = function (dataset, socket) {
    logger.info('RESETTING APP STATE');

    //FIXME explicitly destroy last graph if it exists?

    // ----- BUFFERS (multiplexed over clients) ----------
    //Serve most recent compressed binary buffers
    //TODO reuse across users
    this.lastCompressedVBOs = undefined;
    this.lastMetadata = undefined;
    /** @type {Object.<String,Function>} **/
    this.bufferTransferFinisher = undefined;

    this.lastRenderConfig = undefined;

    //Signal to Explicitly Send New VBOs
    this.updateVboSubject = new Rx.ReplaySubject(1);

    // ----- ANIMATION ------------------------------------
    //current animation
    this.animationStep = driver.create(dataset, socket);
    //multicast of current animation's ticks
    this.ticksMulti = this.animationStep.ticks.publish();
    this.ticksMulti.connect();

    //most recent tick
    this.graph = new Rx.ReplaySubject(1);
    //make available to all clients
    this.ticksMulti.take(1).subscribe(this.graph, log.makeRxErrorHandler(logger, logger, 'ticksMulti failure'));

    logger.trace('RESET APP STATE.');
};

VizServer.prototype.readSelection = function (type, query, res) {
    this.graph.take(1).do(function (graph) {
        graph.simulator.selectNodesInRect(query.sel).then(function (nodeIndices) {
            var edgeIndices = graph.simulator.connectedEdges(nodeIndices);
            return {
                'point': nodeIndices,
                'edge': edgeIndices
            };
        }).then(function (lastSelectionIndices) {
            var page = parseInt(query.page);
            var per_page = parseInt(query.per_page);
            var start = (page - 1) * per_page;
            var end = start + per_page;
            var data = sliceSelection(graph.dataframe, type, lastSelectionIndices[type], start, end,
                                        query.sort_by, query.order === 'asc', query.search);
            res.send(_.extend(data, {
                page: page
            }));
        }).fail(log.makeQErrorHandler(logger, 'read_selection qLastSelectionIndices'));

    }).subscribe(
        _.identity,
        function (err) {
            log.makeRxErrorHandler(logger, 'read_selection handler')(err);
        }
    );
};

VizServer.prototype.tickGraph = function (cb) {
    this.graph.take(1).do(function (graphContent) {
        this.updateVboSubject.onNext(graphContent);
    }.bind(this)).subscribe(
        _.identity,
        function (err) {
            failWithMessage(cb, 'aggregate error');
            log.makeRxErrorHandler(logger, 'aggregate handler')(err);
        }
    );
};

// TODO Extract a graph method and manage graph contexts by filter data operation.
VizServer.prototype.filterGraphByMaskList = function (graph, selectionMasks, exclusionMasks, errors, viewConfig, cb) {
    var response = {filters: viewConfig.filters, exclusions: viewConfig.exclusions};

    var unprunedMasks = graph.dataframe.composeMasks(selectionMasks, exclusionMasks, viewConfig.limits);
    // Prune out dangling edges.
    var masks = graph.dataframe.pruneMaskEdges(unprunedMasks);
    // Prune out orphans if configured that way:
    if (viewConfig.parameters.pruneOrphans === true) {
        masks = graph.dataframe.pruneOrphans(masks);
    }

    // Return early if mask is same as graph's active mask.
    if (masks.equals(graph.dataframe.lastMasks)) {
        var sets = vizSetsToPresentFromViewConfig(viewConfig, graph.dataframe);
        _.extend(response, {success: true, sets:sets});
        cb(response);
        return;
    }

    logger.debug('mask lengths: ', masks.numEdges(), masks.numPoints());

    // Promise
    var simulator = graph.simulator;
    try {
        graph.dataframe.applyDataframeMaskToFilterInPlace(masks, simulator)
            .then(function () {
                simulator.layoutAlgorithms
                    .map(function (alg) {
                        return alg.updateDataframeBuffers(simulator);
                    });
            }).then(function () {
                simulator.tickBuffers([
                    'curPoints', 'pointSizes', 'pointColors',
                    'edgeColors', 'logicalEdges', 'springsPos'
                ]);

                this.tickGraph(cb);
                var sets = vizSetsToPresentFromViewConfig(viewConfig, graph.dataframe);
                _.extend(response, {success: true, sets: sets, errors: errors});
                _.each(errors, logger.debug.bind(logger));
                cb(response);
            }.bind(this)).done(_.identity, function (err) {
                log.makeQErrorHandler(logger, 'dataframe filter')(err);
                errors.push(err);
                _.each(errors, logger.debug.bind(logger));
                _.extend(response, {success: false, errors: errors});
                cb(response);
            });
    } catch (err) {
        log.makeQErrorHandler(logger, 'dataframe filter')(err);
        errors.push(err);
        _.each(errors, logger.debug.bind(logger));
        _.extend(response, {success: false, errors: errors});
        cb(response);
    }
};

function getNamespaceFromGraph(graph) {
    var dataframeColumnsByType = graph.dataframe.getColumnsByType();
    // TODO add special names that can be used in calculation references.
    // TODO handle multiple sources.
    var metadata = _.extend({}, dataframeColumnsByType);
    return metadata;
}

function processAggregateIndices (request, nodeIndices) {
    var graph = request.graph;
    var cb = request.cb;
    var query = request.query;

    logger.debug('Done selecting indices');
    try {
        var edgeIndices = graph.simulator.connectedEdges(nodeIndices);
        var indices = {
            point: nodeIndices,
            edge: edgeIndices
        };
        var data;

        // Initial case of getting global Stats
        // TODO: Make this match the same structure, not the current approach in StreamGL
        if (query.type) {
            data = [
                function () {
                    return graph.dataframe.aggregate(
                        indices[query.type],
                        query.attributes, query.binning, query.mode, query.type);
                }];
        } else {
            var types = ['point', 'edge'];
            data = _.map(types, function (type) {
                var filteredAttributes = _.filter(query.attributes, function (attr) {
                    return (attr.type === type);
                });
                var attributeNames = _.pluck(filteredAttributes, 'name');
                return function () {
                    return graph.dataframe.aggregate(
                        indices[type],
                        attributeNames,
                        query.binning, query.mode, type);
                };
            });
        }

        return util.chainQAll(data).spread(function () {
            var returnData = {};
            _.each(arguments, function (partialData) {
                _.extend(returnData, partialData);
            });
            logger.debug('Sending back aggregate data');
            cb({success: true, data: returnData});
        });

    } catch (err) {
        cb({success: false, error: err.message, stack: err.stack});
        log.makeRxErrorHandler(logger,'aggregate inner handler')(err);
    }
}

/**
 * @param {Object} viewConfig
 * @param {Dataframe} dataframe
 * @returns {Object[]}
 */
function vizSetsToPresentFromViewConfig (viewConfig, dataframe) {
    var sets = viewConfig.sets;
    _.each(sets, function (vizSet) {
        switch (vizSet.id) {
            case 'dataframe':
                vizSet.masks = dataframe.fullDataframeMask();
                break;
            case 'filtered':
                vizSet.masks = dataframe.lastMasks;
                break;
            case 'selection':
                vizSet.masks = dataframe.lastSelectionMasks;
                break;
        }
    });
    return _.map(sets, function (vizSet) { return dataframe.presentVizSet(vizSet); });
}

var setPropertyWhiteList = ['title', 'description'];

function updateVizSetFromClientSet (matchingSet, updatedVizSet) {
    _.extend(matchingSet, _.pick(updatedVizSet, setPropertyWhiteList));
    matchingSet.masks.fromJSON(updatedVizSet.masks);
}

function failWithMessage (cb, message) {
    cb({success: false, error: message});
}

function VizServer(app, socket, cachedVBOs) {
    logger.info('Client connected', socket.id);

    this.isActive = true;
    this.defineRoutesInApp(app);
    this.socket = socket;
    this.cachedVBOs = cachedVBOs;
    /** @type {GraphistryURLParams} */
    var query = this.socket.handshake.query;
    this.viewConfig = new Rx.BehaviorSubject(workbook.blankViewTemplate);
    this.workbookDoc = new Rx.ReplaySubject(1);
    this.workbookForQuery(this.workbookDoc, query);
    this.workbookDoc.subscribe(function (workbookDoc) {
        this.viewConfig.onNext(this.getViewToLoad(workbookDoc, query));
    }.bind(this), log.makeRxErrorHandler(logger, 'Getting View from Workbook'));

    this.setupColorTexture();

    var renderConfigDeferred = Q.defer();
    this.qRenderConfig = renderConfigDeferred.promise;
    this.workbookDoc.take(1).do(function (workbookDoc) {
        this.qDataset = this.setupDataset(workbookDoc, query);
        this.qDataset.then(function (dataset) {
            var metadata = dataset.metadata;

            if (!(metadata.scene in rConf.scenes)) {
                logger.warn('WARNING Unknown scene "%s", using default', metadata.scene);
                metadata.scene = 'default';
            }

            this.resetState(dataset, socket);
            renderConfigDeferred.resolve(rConf.scenes[metadata.scene]);
        }.bind(this)).fail(log.makeQErrorHandler(logger, 'resetting state'));
    }.bind(this)).subscribe(_.identity, log.makeRxErrorHandler(logger, 'Get render config'));

    this.socket.on('update_view_parameter', function (spec, cb) {
        this.viewConfig.take(1).do(function (viewConfig) {
            viewConfig.parameters[spec.name] = spec.value;
            cb({success: true});
        }).subscribe(_.identity, function (err) {
            cb({success: false, errors: [err.message]});
            log.makeRxErrorHandler(logger, 'Update view parameter')(err);
        });
    }.bind(this));

    this.socket.on('render_config', function(_, cb) {
        this.qRenderConfig.then(function (renderConfig) {
            logger.info('renderConfig', renderConfig);
            logger.trace('Sending render-config to client');
            cb({success: true, renderConfig: renderConfig});

            if (saveAtEachStep) {
                persist.saveConfig(defaultSnapshotName, renderConfig);
            }

            this.lastRenderConfig = renderConfig;
        }.bind(this)).fail(function (err) {
            failWithMessage(cb, 'Render config read error');
            log.makeQErrorHandler(logger, 'sending render_config')(err);
        });
    }.bind(this));

    this.socket.on('update_render_config', function(newValues, cb) {
        this.qRenderConfig.then(function (renderConfig) {
            logger.info('renderConfig [before]', renderConfig);
            logger.trace('Updating render-config from client values');

            extend(true, renderConfig, newValues);

            cb({success: true, renderConfig: renderConfig});

            if (saveAtEachStep) {
                persist.saveConfig(defaultSnapshotName, renderConfig);
            }

            this.lastRenderConfig = renderConfig;
        }.bind(this)).fail(function (err) {
            failWithMessage(cb, 'Render config update error');
            log.makeQErrorHandler(logger, 'updating render_config')(err);
        });
    }.bind(this));

    /**
     * @typedef {Object} Point2D
     * @property {Number} x
     * @property {Number} y
     */

    /**
     * @typedef {Object} Rect
     * @property {Point2D} tl top left corner
     * @property {Point2D} br bottom right corner
     */

    /**
     * @typedef {Object} Circle
     * @property {Point2D} center
     * @property {Number} radius
     */

    /**
     * @typedef {Object} SetSpecification
     * @property {String} sourceType one of selection,dataframe,filtered
     * @property {Rect} sel rectangle/etc selection gesture.
     * @property {Circle} circle
     */

    this.socket.on('create_set', function (sourceType, specification, cb) {
        /**
         * @type {SetSpecification} specification
         */
        Rx.Observable.combineLatest(this.graph, this.viewConfig, function (graph, viewConfig) {
            var qNodeSelection;
            var pointsOnly = false;
            var dataframe = graph.dataframe;
            var simulator = graph.simulator;
            if (sourceType === 'selection' || sourceType === undefined) {
                var clientMaskSet = specification.masks;
                if (specification.sel !== undefined) {
                    var rect = specification.sel;
                    pointsOnly = true;
                    qNodeSelection = simulator.selectNodesInRect(rect);
                } else if (specification.circle !== undefined) {
                    var circle = specification.circle;
                    pointsOnly = true;
                    qNodeSelection = simulator.selectNodesInCircle(circle);
                } else if (clientMaskSet !== undefined) {
                    // translate client masks to rawdata masks.
                    qNodeSelection = Q(new DataframeMask(dataframe, clientMaskSet.point, clientMaskSet.edge, dataframe.lastMasks));
                } else {
                    throw Error('Selection not specified for creating a Set');
                }
                if (pointsOnly) {
                    qNodeSelection = qNodeSelection.then(function (pointIndexes) {
                        var edgeIndexes = simulator.connectedEdges(pointIndexes);
                        return new DataframeMask(dataframe, pointIndexes, edgeIndexes);
                    });
                }
            } else if (sourceType === 'dataframe') {
                qNodeSelection = Q(dataframe.fullDataframeMask());
            } else if (sourceType === 'filtered') {
                qNodeSelection = Q(dataframe.lastMasks);
            } else {
                throw Error('Unrecognized special type for creating a Set: ' + sourceType);
            }
            qNodeSelection.then(function (dataframeMask) {
                var newSet = {
                    id: new TransactionalIdentifier().toString(),
                    sourceType: sourceType,
                    specification: _.omit(specification, setPropertyWhiteList),
                    masks: dataframeMask,
                    sizes: {point: dataframeMask.numPoints(), edge: dataframeMask.numEdges()}
                };
                updateVizSetFromClientSet(newSet, specification);
                viewConfig.sets.push(newSet);
                dataframe.masksForVizSets[newSet.id] = dataframeMask;
                cb({success: true, set: dataframe.presentVizSet(newSet)});
            }).fail(log.makeQErrorHandler(logger, 'pin_selection_as_set'));
        }).take(1).subscribe(_.identity,
            function (err) {
                logger.error(err, 'Error creating set from selection');
                failWithMessage(cb, 'Server error when saving the selection as a Set');
            });
    }.bind(this));

    var specialSetKeys = ['dataframe', 'filtered', 'selection'];

    this.socket.on('get_sets', function (cb) {
        logger.trace('sending current sets to client');
        Rx.Observable.combineLatest(this.graph, this.viewConfig, function (graph, viewConfig) {
            var outputSets = vizSetsToPresentFromViewConfig(viewConfig, graph.dataframe);
            cb({success: true, sets: outputSets});
        }.bind(this)).take(1).subscribe(_.identity,
            function (err) {
                logger.error(err, 'Error retrieving Sets');
                failWithMessage(cb, 'Server error when retrieving all Set definitions');
            });
    }.bind(this));

    /**
     * This handles creates (set given with no id), updates (id and set given), and deletes (id with no set).
     */
    this.socket.on('update_set', function (id, updatedVizSet, cb) {
        Rx.Observable.combineLatest(this.graph, this.viewConfig, function (graph, viewConfig) {
            if (_.contains(specialSetKeys, id)) {
                throw Error('Cannot update the special Sets');
            }
            var matchingSetIndex = _.findIndex(viewConfig.sets, function (vizSet) { return vizSet.id === id; });
            if (matchingSetIndex === -1) {
                // Auto-create:
                if (!updatedVizSet) {
                    updatedVizSet = {};
                }
                // Auto-create an ID:
                if (updatedVizSet.id === undefined) {
                    updatedVizSet.id = (id || new TransactionalIdentifier()).toString();
                }
                viewConfig.sets.push(updatedVizSet);
            } else if (updatedVizSet) {
                if (updatedVizSet.id === undefined) {
                    updatedVizSet.id = id;
                }
                var matchingSet = viewConfig.sets[matchingSetIndex];
                updateVizSetFromClientSet(matchingSet, updatedVizSet);
                updatedVizSet = matchingSet;
            } else { // No set given means to delete by id
                viewConfig.sets.splice(matchingSetIndex, 1);
                graph.dataframe.masksForVizSets[id] = undefined;
            }
            cb({success: true, set: graph.dataframe.presentVizSet(updatedVizSet)});
        }).take(1).subscribe(_.identity,
            function (err) {
                logger.error(err, 'Error sending update_set');
                failWithMessage(cb, 'Server error when updating a Set');
                throw err;
            });
    }.bind(this));

    this.socket.on('get_filters', function (cb) {
        logger.trace('sending current filters and exclusions to client');
        this.viewConfig.take(1).do(function (viewConfig) {
            cb({success: true, filters: viewConfig.filters, exclusions: viewConfig.exclusions});
        }).subscribe(
            _.identity, log.makeRxErrorHandler(logger, 'get_filters handler'));
    }.bind(this));

    this.socket.on('update_filters', function (definition, cb) {
        logger.trace('updating filters from client values');
        // Maybe direct assignment isn't safe, but it'll do for now.
        this.viewConfig.take(1).do(function (viewConfig) {
            var bumpViewConfig = false;

            // Update exclusions:
            if (definition.exclusions !== undefined &&
                !_.isEqual(definition.exclusions, viewConfig.exclusions)) {
                viewConfig.exclusions = definition.exclusions;
                bumpViewConfig = true;
            }
            logger.info('updated exclusions', viewConfig.exclusions);

            // Update filters:
            if (definition.filters !== undefined &&
                !_.isEqual(definition.filters, viewConfig.filters)) {
                viewConfig.filters = definition.filters;
                bumpViewConfig = true;
            }
            logger.info('updated filters', viewConfig.filters);

            if (viewConfig.limits === undefined) {
                viewConfig.limits = {point: Infinity, edge: Infinity};
            }

            if (bumpViewConfig) { this.viewConfig.onNext(viewConfig); }

            this.graph.take(1).do(function (graph) {
                var dataframe = graph.dataframe;
                var selectionMasks = [];
                var errors = [];
                var generator = new ExpressionCodeGenerator('javascript');
                var query;

                /** @type {DataframeMask[]} */
                var exclusionMasks = [];
                _.each(viewConfig.exclusions, function (exclusion) {
                    if (exclusion.enabled === false) {
                        return;
                    }
                    /** @type ClientQuery */
                    query = exclusion.query;
                    if (query === undefined) {
                        return;
                    }
                    if (!query.type) {
                        query.type = exclusion.type;
                    }
                    if (!query.attribute) {
                        query.attribute = exclusion.attribute;
                    }
                    var masks = dataframe.getMasksForQuery(query, errors);
                    if (masks !== undefined) {
                        masks.setExclusive(true);
                        exclusion.maskSizes = masks.maskSize();
                        exclusionMasks.push(masks);
                    }
                });

                _.each(viewConfig.filters, function (filter) {
                    if (filter.enabled === false) {
                        return;
                    }
                    /** @type ClientQuery */
                    query = filter.query;
                    if (query === undefined) {
                        return;
                    }
                    var ast = query.ast;
                    if (ast !== undefined &&
                        ast.type === 'Limit' &&
                        ast.value !== undefined) {
                        viewConfig.limits.point = generator.evaluateExpressionFree(ast.value);
                        viewConfig.limits.edge = viewConfig.limits.point;
                        return;
                    }
                    if (!query.type) {
                        query.type = filter.type;
                    }
                    if (!query.attribute) {
                        query.attribute = filter.attribute;
                    }
                    var masks = dataframe.getMasksForQuery(query, errors);
                    if (masks !== undefined) {
                        // Record the size of the filtered set for UI feedback:
                        filter.maskSizes = masks.maskSize();
                        selectionMasks.push(masks);
                    }
                });

                this.filterGraphByMaskList(graph, selectionMasks, exclusionMasks, errors, viewConfig, cb);
            }.bind(this)).subscribe(
                _.identity,
                function (err) {
                    log.makeRxErrorHandler(logger, 'update_filters handler')(err);
                }
            );
        }.bind(this)).subscribe(_.identity, log.makeRxErrorHandler(logger, 'get_filters handler'));
    }.bind(this));

    this.socket.on('layout_controls', function(_, cb) {
        logger.info('Sending layout controls to client');

        this.graph.take(1).do(function (graph) {
            logger.info('Got layout controls');
            var controls = graph.simulator.controls;
            cb({success: true, controls: lConf.toClient(controls.layoutAlgorithms)});
        })
        .subscribeOnError(function (err) {
            logger.error(err, 'Error sending layout_controls');
            failWithMessage(cb, 'Server error when fetching controls');
            throw err;
        });
    }.bind(this));

    this.socket.on('begin_streaming', function(_, cb) {
        this.qRenderConfig.then(function (renderConfig) {
            this.beginStreaming(renderConfig, this.colorTexture);
            if (cb) {
                cb({success: true});
            }
        }.bind(this)).fail(log.makeQErrorHandler(logger, 'begin_streaming'));
    }.bind(this));

    this.socket.on('reset_graph', function (_, cb) {
        logger.info('reset_graph command');
        this.qDataset.then(function (dataset) {
            this.resetState(dataset, this.socket);
            cb();
        }.bind(this)).fail(log.makeQErrorHandler(logger, 'reset_graph request'));
    }.bind(this));

    this.socket.on('inspect_header', function (nothing, cb) {
        logger.info('inspect header');
        this.graph.take(1).do(function (graph) {
            cb({
                success: true,
                header: {
                    nodes: graph.dataframe.getAttributeKeys('point'),
                    edges: graph.dataframe.getAttributeKeys('edge')
                },
                urns: {
                    nodes: 'read_node_selection',
                    edges: 'read_edge_selection'
                }
            });
        }).subscribe(
            _.identity,
            function (err) {
                failWithMessage(cb, 'inspect_header error');
                log.makeRxErrorHandler(logger, 'inspect_header handler')(err);
            }
        );
    }.bind(this));

    /** Implements/gets a namespace comprehension, for calculation references and metadata. */
    this.socket.on('get_namespace_metadata', function (cb) {
        logger.trace('Sending Namespace metadata to client');
        this.graph.take(1).do(function (graph) {
            var metadata = getNamespaceFromGraph(graph);
            cb({success: true,
                metadata: metadata});
        }).subscribe(
            _.identity,
            function (err) {
                failWithMessage(cb, 'Namespace metadata error');
                log.makeQErrorHandler(logger, 'sending namespace metadata')(err);
            }
        );
    }.bind(this));

    this.socket.on('update_namespace_metadata', function (updates, cb) {
        logger.trace('Updating Namespace metadata from client');
        this.graph.take(1).do(function (graph) {
            var metadata = getNamespaceFromGraph(graph);
            // set success to true when we support update and it succeeds:
            cb({success: false, metadata: metadata});
        }).fail(function (/*err*/) {
            failWithMessage(cb, 'Namespace metadata update error');
            log.makeQErrorHandler(logger, 'updating namespace metadata');
        });
    }.bind(this));

    // Legacy method for timeslider.js only; refactor that to work with newer code and kill this.
    this.socket.on('filter', function (query, cb) {
        logger.info('Got filter', query);
        Rx.Observable.combineLatest(this.viewConfig, this.graph, function (viewConfig, graph) {

            var selectionMasks = [];
            var errors = [];

            var dataframe = graph.dataframe;
            _.each(query, function (data, attribute) {
                var type = data.type;
                var normalization = dataframe.normalizeAttributeName(attribute, type);
                if (normalization === undefined) {
                    errors.push(Error('No attribute found for: ' + attribute + ',' + type));
                    cb({success: false, errors: errors});
                    return;
                } else {
                    type = normalization.type;
                    attribute = normalization.attribute;
                }
                try {
                    var filterFunc = dataframe.filterFuncForQueryObject(data);
                    var masks = dataframe.getAttributeMask(type, attribute, filterFunc);
                    // Record the size of the filtered set for UI feedback:
                    filter.maskSizes = masks.maskSize();
                    selectionMasks.push(masks);
                } catch (e) {
                    errors.push(e.message);
                }
            });
            this.filterGraphByMaskList(graph, selectionMasks, undefined, errors, viewConfig, cb);
        }.bind(this)).take(1).subscribe(
            _.identity,
            function (err) {
                log.makeRxErrorHandler(logger, 'filter handler')(err);
            }
        );
    }.bind(this));

    this.socket.on('encode_by_column', function (query, cb) {
        this.graph.take(1).do(function (graph) {
            var dataframe = graph.dataframe,
                normalization = dataframe.normalizeAttributeName(query.attribute, query.type),
                encodingType = query.encodingType,
                variation = query.variation,
                binning = query.binning;
            if (normalization === undefined) {
                failWithMessage(cb, 'No attribute found for: ' + query.attribute + ',' + query.type);
                return;
            }
            var attributeName = normalization.attribute,
                type = normalization.type;
            if (encodingType) {
                if (encodingType === 'color' || encodingType === 'size' || encodingType === 'opacity') {
                    encodingType = type + encodingType.charAt(0).toLocaleUpperCase() + encodingType.slice(1);
                }
                if (encodingType.indexOf(type) !== 0) {
                    failWithMessage(cb, 'Attribute type does not match encoding type requested.');
                    return;
                }
            }
            var encoding;
            try {
                encoding = encodings.inferEncoding(dataframe, type, attributeName, encodingType, variation, binning);
            } catch (e) {
                failWithMessage(cb, e.message);
                return;
            }
            if (encoding === undefined || encoding.scaling === undefined) {
                failWithMessage(cb, 'No scaling inferred for: ' + encodingType);
                return;
            }
            var bufferName = encoding.bufferName;
            var enabled = false;
            if (query.reset) {
                dataframe.resetLocalBuffer(bufferName);
            } else {
                var sourceValues = dataframe.getColumnValues(attributeName, type);
                var encodedAttributeName = bufferName + '_' + attributeName;
                var encodedColumnValues;
                // TODO fix how we map to and recolor edges.
                if (bufferName === 'edgeColors') {
                    encodedColumnValues = new Array(sourceValues.length * 2);
                    for (var i=0; i<sourceValues.length; i++) {
                        var value = sourceValues[i];
                        encodedColumnValues[2*i] = encoding.scaling(value);
                        encodedColumnValues[2*i+1] = encoding.scaling(value);
                    }
                } else {
                    encodedColumnValues = _.map(sourceValues, encoding.scaling);
                }
                dataframe.overlayLocalBuffer(bufferName, encodedAttributeName, encodedColumnValues);
                enabled = true;
            }
            graph.simulator.tickBuffers([bufferName]);
            this.animationStep.interact({play: true, layout: false});
            cb(_.extend({success: true, enabled: enabled}, _.omit(encoding, ['scaling'])));
        }.bind(this)).subscribe(
            _.identity,
            function (err) {
                log.makeRxErrorHandler(logger, 'recolor by column handler')(err);
            }
        );
    }.bind(this));

    this.setupAggregationRequestHandling();

    this.socket.on('viz', function (msg, cb) { cb({success: true}); });
}

/** Pick the view to load for this query.
 * @param {Object} workbookDoc
 * @param {GraphistryURLParams} query
 * @returns {Object}
 */
VizServer.prototype.getViewToLoad = function (workbookDoc, query) {
    // Pick the default view or the current view or any view.
    var viewConfig = workbookDoc.views.default ||
        (workbookDoc.currentView ?
            workbookDoc.views[workbookDoc.currentview] : _.find(workbookDoc.views));
    // Apply approved URL parameters to that view concretely since we're creating it now:
    _.extend(viewConfig, _.pick(query, workbook.URLParamsThatPersist));
    return viewConfig;
};

/** Get the dataset name from the query parameters, may have been loaded from view:
 * @param {Object} workbookDoc
 * @param {GraphistryURLParams} query
 * @returns {Promise}
 */
VizServer.prototype.setupDataset = function (workbookDoc, query) {
    var queryDatasetURL = loader.datasetURLFromQuery(query),
        queryDatasetConfig = loader.datasetConfigFromQuery(query);
    var datasetURLString, datasetConfig;
    if (queryDatasetURL === undefined) {
        logger.debug('No dataset in URL; picking random in workbook');
        datasetConfig = _.find(workbookDoc.datasetReferences);
        datasetURLString = datasetConfig.url;
    } else {
        // Using the URL parameter, make a config from the URL:
        datasetURLString = queryDatasetURL.format();
        _.extend(queryDatasetConfig, {
            name: datasetURLString,
            url: datasetURLString
        });
    }
    // Auto-create a config for the URL:
    if (!workbookDoc.datasetReferences.hasOwnProperty(datasetURLString)) {
        workbookDoc.datasetReferences[datasetURLString] = {};
    }
    // Select the config and update it from the query unless the URL mismatches:
    datasetConfig = workbookDoc.datasetReferences[datasetURLString];
    if (datasetConfig.url === undefined ||
        queryDatasetURL === undefined ||
        datasetConfig.url === datasetURLString) {
        _.extend(datasetConfig, queryDatasetConfig);
    }

    // Pass the config on:
    return loader.downloadDataset(datasetConfig);
};

VizServer.prototype.workbookForQuery = function (observableResult, query) {
    if (query.workbook) {
        logger.debug('Loading workbook', query.workbook);
        workbook.loadDocument(decodeURIComponent(query.workbook)).subscribe(function (workbookDoc) {
            observableResult.onNext(workbookDoc);
        }, function (error) {
            log.makeRxErrorHandler(logger, 'Loading Workbook')(error);
            // TODO report to user if authenticated and can know of this workbook's existence.
        });
    } else {
        // Create a new workbook here with a default view:
        observableResult.onNext(workbook.blankWorkbookTemplate);
    }
};

VizServer.prototype.setupColorTexture = function () {
    this.colorTexture = new Rx.ReplaySubject(1);
    var imgPath = path.resolve(__dirname, 'test-colormap2.rgba');
    var img =
        Rx.Observable.fromNodeCallback(fs.readFile)(imgPath)
            .flatMap(function (buffer) {
                logger.trace('Loaded raw colorTexture', buffer.length);
                return Rx.Observable.fromNodeCallback(compress.deflate)(
                    buffer,//binary,
                    {output: new Buffer(
                        Math.max(1024, Math.round(buffer.length * 1.5)))})
                    .map(function (compressed) {
                        return {
                            raw: buffer,
                            compressed: compressed
                        };
                    });
            })
            .do(function () { logger.trace('Compressed color texture'); })
            .map(function (pair) {
                logger.trace('colorMap bytes', pair.raw.length);
                return {
                    buffer: pair.compressed[0],
                    bytes: pair.raw.length,
                    width: 512,
                    height: 512
                };
            });

    img.take(1)
        .do(this.colorTexture)
        .subscribe(_.identity, log.makeRxErrorHandler(logger, 'img/texture'));
    this.colorTexture
        .do(function() { logger.trace('HAS COLOR TEXTURE'); })
        .subscribe(_.identity, log.makeRxErrorHandler(logger, 'colorTexture'));
};

VizServer.prototype.setupAggregationRequestHandling = function () {
    var aggregateRequests = new Rx.Subject().controlled(); // Use pull model.

    //query :: {attributes: ??, binning: ??, mode: ??, type: 'point' + 'edge'}
    // -> {success: false} + {success: true, data: ??}
    this.socket.on('aggregate', function (query, cb) {
        logger.info('Got aggregate', query);

        this.graph.take(1).do(function (graph) {
            logger.trace('Selecting Indices');
            var qIndices;

            if (query.all === true) {
                var numPoints = graph.simulator.dataframe.getNumElements('point');
                qIndices = Q(new Uint32Array(_.range(numPoints)));
            } else if (!query.sel) {
                qIndices = Q(new Uint32Array([]));
            } else {
                qIndices = graph.simulator.selectNodesInRect(query.sel);
            }

            aggregateRequests.subject.onNext({
                qIndices: qIndices,
                graph: graph,
                query: query,
                cb: cb
            });

        }).subscribe(
            _.identity,
            function (err) {
                failWithMessage(cb, 'aggregate socket error');
                log.makeRxErrorHandler(logger, 'aggregate socket handler')(err);
            }
        );
    }.bind(this));

    // Handle aggregate requests. Fully handle one before moving on to the next.
    aggregateRequests.do(function (request) {
        request.qIndices.then(processAggregateIndices.bind(null, request))
            .then(function () {
                aggregateRequests.request(1);
            }).done(_.identity, log.makeQErrorHandler(logger, 'AggregateIndices Q'));
    }).subscribe(_.identity, log.makeRxErrorHandler(logger, 'aggregate request loop'));
    aggregateRequests.request(1); // Always request first.
};

// FIXME: ExpressJS routing does not support re-targeting. So we set a global for now!
var appRouteResponder;

VizServer.prototype.defineRoutesInApp = function (app) {
    this.app = app;

    var routesAlreadyBound = (appRouteResponder !== undefined);
    appRouteResponder = this;
    if (routesAlreadyBound) { return; }

    this.app.get('/vbo', function (req, res) {
        logger.info('VBOs: HTTP GET %s', req.originalUrl);
        // performance monitor here?
        // profiling.debug('VBO request');

        try {
            // TODO: check that query parameters are present, and that given id, buffer exist
            var bufferName = req.query.buffer;
            var id = req.query.id;

            res.set('Content-Encoding', 'gzip');
            var VBOs = (id === appRouteResponder.socket.id ? appRouteResponder.lastCompressedVBOs : appRouteResponder.cachedVBOs[id]);
            if (VBOs) {
                res.send(VBOs[bufferName]);
            }
            res.send();

            var bufferTransferFinisher = appRouteResponder.bufferTransferFinisher;
            if (bufferTransferFinisher) {
                bufferTransferFinisher(bufferName);
            }
        } catch (e) {
            log.makeQErrorHandler(logger, 'bad /vbo request')(e);
        }
    });

    this.app.get('/texture', function (req, res) {
        logger.debug('got texture req', req.originalUrl, req.query);
        try {
            appRouteResponder.colorTexture.pluck('buffer').do(
                function (data) {
                    res.set('Content-Encoding', 'gzip');
                    res.send(data);
                })
                .subscribe(_.identity, log.makeRxErrorHandler(logger, 'colorTexture pluck'));

        } catch (e) {
            log.makeQErrorHandler(logger, 'bad /texture request')(e);
        }
    });

    this.app.get('/read_node_selection', function (req, res) {
        logger.debug('Got read_node_selection', req.query);

        // HACK because we're sending numbers across a URL string parameter.
        // This should be sent in a type aware manner
        if (req.query.sel.br) {
            var sel = req.query.sel;
            sel.br.x = +sel.br.x;
            sel.br.y = +sel.br.y;
            sel.tl.x = +sel.tl.x;
            sel.tl.y = +sel.tl.y;
        }

        appRouteResponder.readSelection('point', req.query, res);
    });

    this.app.get('/read_edge_selection', function (req, res) {
        logger.debug('Got read_edge_selection', req.query);

        // HACK because we're sending numbers across a URL string parameter.
        // This should be sent in a type aware manner
        if (req.query.sel.br) {
            var sel = req.query.sel;
            sel.br.x = +sel.br.x;
            sel.br.y = +sel.br.y;
            sel.tl.x = +sel.tl.x;
            sel.tl.y = +sel.tl.y;
        }

        appRouteResponder.readSelection('edge', req.query, res);
    });
};

VizServer.prototype.rememberVBOs = function (VBOs) {
    this.lastCompressedVBOs = VBOs;
    this.cachedVBOs[this.socket.id] = this.lastCompressedVBOs;
};

VizServer.prototype.beginStreaming = function (renderConfig, colorTexture) {

    // ========== BASIC COMMANDS
    this.rememberVBOs({});
    this.socket.on('disconnect', function () {
        this.dispose();
    }.bind(this));

    //Used for tracking what needs to be sent
    //Starts as all active, and as client caches, whittles down
    var activeBuffers = _.chain(renderConfig.models).pairs().filter(function (pair) {
        var model = pair[1];
        return rConf.isBufServerSide(model);
    }).map(function (pair) {
        return pair[0];
    }).value();

    var activeTextures = _.chain(renderConfig.textures).pairs().filter(function (pair) {
        var texture = pair[1];
        return rConf.isTextureServerSide(texture);
    }).map(function (pair) {
        return pair[0];
    }).value();

    var activePrograms = renderConfig.render;



    var requestedBuffers = activeBuffers,
        requestedTextures = activeTextures;

    //Knowing this helps overlap communication and computations
    this.socket.on('planned_binary_requests', function (request) {
        logger.debug('CLIENT SETTING PLANNED REQUESTS', request.buffers, request.textures);
        requestedBuffers = request.buffers;
        requestedTextures = request.textures;
    });


    logger.debug('active buffers/textures/programs', activeBuffers, activeTextures, activePrograms);

    var graph = this.graph;
    var animationStep = this.animationStep;

    this.socket.on('interaction', function (payload) {
        // performance monitor here?
        // profiling.trace('Got Interaction');
        logger.trace('Got interaction:', payload);
        // TODO: Find a way to avoid flooding main thread waiting for GPU ticks.
        var defaults = {play: false, layout: false};
        animationStep.interact(_.extend(defaults, payload || {}));
    });

    this.socket.on('get_labels', function (query, cb) {

        var indices = query.indices;
        var dim = query.dim;

        graph.take(1)
            .map(function (graph) {
                return labeler.getLabels(graph, indices, dim);
            })
            .do(function (out) {
                cb(null, out);
            })
            .subscribe(
                _.identity,
                function (err) {
                    cb('get_labels error');
                    log.makeRxErrorHandler(logger, 'get_labels')(err);
                });
    });

    this.socket.on('get_global_ids', function (sel, cb) {
        graph.take(1).do(function (graph) {
            var res = _.map(sel, function (ent) {
                var type = ent.dim === 1 ? 'point' : 'edge';
                return {
                    type: type,
                    dataIdx: graph.simulator.dataframe.globalize(ent.idx, type),
                    viewIdx: ent.idx
                };
            });
            cb({success: true, ids: res});
        }).subscribe(_.identity, log.makeRxErrorHandler(logger, 'get_global_ids'));
    });

    this.socket.on('shortest_path', function (pair) {
        graph.take(1)
            .do(function (graph) {
                graph.simulator.highlightShortestPaths(pair);
                animationStep.interact({play: true, layout: false});
            })
            .subscribe(_.identity, log.makeRxErrorHandler(logger, 'shortest_path'));
    });

    this.socket.on('set_colors', function (color) {
        graph.take(1)
            .do(function (graph) {
                graph.simulator.setColor(color);
                animationStep.interact({play: true, layout: false});
            })
            .subscribe(_.identity, log.makeRxErrorHandler(logger, 'set_colors'));
    });

    /**
     * @typedef {Object} SelectionSpecification
     * @property {String} action add/remove/replace
     * @property {String} gesture rectangle/circle/masks
     */

    /** This represents a single selection action.
     */
    this.socket.on('select', function (specification, cb) {
        /** @type {SelectionSpecification} specification */
        Rx.Observable.combineLatest(this.graph, this.viewConfig, function (graph, viewConfig) {
            var qNodeSelection;
            var simulator = graph.simulator;
            switch (specification.gesture) {
                case 'rectangle':
                    qNodeSelection = simulator.selectNodesInRect({sel: _.pick(specification, ['tl', 'br'])});
                    break;
                case 'circle':
                    qNodeSelection = simulator.selectNodesInCircle(_.pick(specification, ['center', 'radius']));
                    break;
                case 'masks':
                    // TODO FIXME translate masks to unfiltered indexes.
                    qNodeSelection = Q(specification.masks);
                    break;
                case 'sets':
                    var matchingSets = _.filter(viewConfig.sets, function (vizSet) {
                        return specification.setIDs.indexOf(vizSet.id) !== -1;
                    });
                    var combinedMasks = _.reduce(matchingSets, function (masks, vizSet) {
                        return masks.union(vizSet.masks);
                    }, new DataframeMask(graph.dataframe, [], []));
                    qNodeSelection = Q(combinedMasks);
                    break;
                default:
                    throw Error('Unrecognized selection gesture: ' + specification.gesture.toString());
            }
            if (qNodeSelection === undefined) { throw Error('No selection made'); }
            var lastMasks = graph.dataframe.lastSelectionMasks;
            switch (specification.action) {
                case 'add':
                    qNodeSelection = qNodeSelection.then(function (dataframeMask) {
                        return lastMasks.union(dataframeMask);
                    });
                    break;
                case 'remove':
                    qNodeSelection = qNodeSelection.then(function (dataframeMask) {
                        return lastMasks.minus(dataframeMask);
                    });
                    break;
                case 'replace':
                    break;
                default:
                    break;
            }
            qNodeSelection.then(function (dataframeMask) {
                graph.dataframe.lastSelectionMasks = dataframeMask;
                graph.simulator.tickBuffers(['selectedPointIndexes', 'selectedEdgeIndexes']);
                animationStep.interact({play: true, layout: false});
                cb({success: true});
            });
        }).take(1).subscribe(_.identity,
            function (err) {
                logger.error(err, 'Error modifying the selection');
                failWithMessage(cb, 'Server error when modifying the selection');
            });
    }.bind(this));

    this.socket.on('highlight', function (specification, cb) {
        /** @type {SelectionSpecification} specification */
        Rx.Observable.combineLatest(this.graph, this.viewConfig, function (graph, viewConfig) {
            var qNodeSelection;
            switch (specification.gesture) {
                case 'masks':
                    // TODO FIXME translate masks to unfiltered indexes.
                    qNodeSelection = Q(specification.masks);
                    break;
                case 'sets':
                    var matchingSets = _.filter(viewConfig.sets, function (vizSet) {
                        return specification.setIDs.indexOf(vizSet.id) !== -1;
                    });
                    var combinedMasks = _.reduce(matchingSets, function (masks, vizSet) {
                        return masks.union(vizSet.masks);
                    }, new DataframeMask(graph.dataframe, [], []));
                    qNodeSelection = Q(combinedMasks);
                    break;
                default:
                    throw Error('Unrecognized highlight gesture: ' + specification.gesture.toString());
            }
            var GREEN = 255 << 8;
            var color = specification.color || GREEN;
            qNodeSelection.then(function (dataframeMask) {
                var simulator = graph.simulator, dataframe = simulator.dataframe;
                var bufferName = 'pointColors';
                if (dataframeMask.isEmpty() && dataframe.canResetLocalBuffer(bufferName)) {
                    dataframe.resetLocalBuffer(bufferName);
                } else {
                    var pointColorsBuffer = dataframe.getLocalBuffer(bufferName);
                    var highlightedPointColorsBuffer = _.clone(pointColorsBuffer);
                    dataframeMask.mapPointIndexes(function (pointIndex) {
                        highlightedPointColorsBuffer[pointIndex] = color;
                    });
                }
                simulator.tickBuffers([bufferName]);

                animationStep.interact({play: true, layout: false});
                cb({success: true});
            });
        }).take(1).subscribe(_.identity,
            function (err) {
                logger.error(err, 'Error performing a highlight');
                failWithMessage(cb, 'Server error when performing a highlight');
            });
    }.bind(this));

    this.socket.on('highlight_points', function (points) {
        graph.take(1)
            .do(function (graph) {

                points.forEach(function (point) {
                    graph.simulator.dataframe.getLocalBuffer('pointColors')[point.index] = point.color;
                    // graph.simulator.buffersLocal.pointColors[point.index] = point.color;
                });
                graph.simulator.tickBuffers(['pointColors']);

                animationStep.interact({play: true, layout: false});
            })
            .subscribe(_.identity, log.makeRxErrorHandler(logger, 'highlighted_points'));

    });

    this.socket.on('persist_current_workbook', function(workbookName, cb) {
        Rx.Observable.combineLatest(graph, this.workbookDoc, function (graph, workbookDoc) {
            workbookDoc.title = workbookName;
            workbookDoc.contentName = workbookName;
            workbook.saveDocument(workbookName, workbookDoc).then(
                function (result) {
                    return cb({success: true, data: result});
                },
                function (rejectedResult) {
                    return failWithMessage(cb, rejectedResult);
                });
            }).take(1).subscribe(_.identity, log.makeRxErrorHandler(logger, 'persist_current_workbook'));
    }.bind(this));

    this.socket.on('persist_current_vbo', function(contentKey, cb) {
        graph.take(1)
            .do(function (graph) {
                var cleanContentKey = encodeURIComponent(contentKey);
                persist.publishStaticContents(
                    cleanContentKey, this.lastCompressedVBOs,
                    this.lastMetadata, graph.dataframe, renderConfig
                ).then(function() {
                    cb({success: true, name: cleanContentKey});
                }).catch(function (error) {
                    cb({success: false, name: cleanContentKey});
                }).done(
                    _.identity,
                    log.makeQErrorHandler(logger, 'persist_current_vbo')
                );
            }.bind(this))
            .subscribe(_.identity, log.makeRxErrorHandler(logger, 'persist_current_vbo'));
    }.bind(this));

    this.socket.on('persist_upload_png_export', function(pngDataURL, contentKey, imageName, cb) {
        imageName = imageName || 'preview.png';
        graph.take(1)
            .do(function (/*graph*/) {
                var cleanContentKey = encodeURIComponent(contentKey),
                    cleanImageName = encodeURIComponent(imageName),
                    base64Data = pngDataURL.replace(/^data:image\/png;base64,/,""),
                    binaryData = new Buffer(base64Data, 'base64');
                persist.publishPNGToStaticContents(cleanContentKey, cleanImageName, binaryData).then(function() {
                    cb({success: true, name: cleanContentKey});
                }).done(
                    _.identity,
                    log.makeQErrorHandler(logger, 'persist_upload_png_export')
                );
            })
            .subscribe(_.identity, log.makeRxErrorHandler(logger, 'persist_upload_png_export'));
    });

    this.socket.on('fork_vgraph', function (name, cb) {
        graph.take(1)
            .do(function (graph) {
                var vgName = 'Users/' + encodeURIComponent(name);
                vgwriter.save(graph, vgName).then(function () {
                    cb({success: true, name: vgName});
                }).done(
                    _.identity,
                    log.makeQErrorHandler(logger, 'fork_vgraph')
                );
            })
            .subscribe(_.identity, function (err) {
                failWithMessage(cb, 'fork_vgraph error');
                log.makeRxErrorHandler(logger, 'fork_vgraph error')(err);
            });
    });






    // ============= EVENT LOOP

    //starts true, set to false whenever transfer starts, true again when acknowledged.
    var clientReady = new Rx.ReplaySubject(1);
    clientReady.onNext(true);
    this.socket.on('received_buffers', function (time) {
        perf.gauge('graph-viz:driver:viz-server, client end-to-end time', time);
        logger.trace('Client end-to-end time', time);
        clientReady.onNext(true);
    });

    clientReady.subscribe(logger.debug.bind(logger, 'CLIENT STATUS'), log.makeRxErrorHandler(logger, 'clientReady'));

    logger.trace('SETTING UP CLIENT EVENT LOOP ===================================================================');
    var step = 0;
    var lastVersions = null;
    var lastTick = 0;

    var graphObservable = graph;
    graph.expand(function (graph) {
        step++;

        var ticker = {step: step};

        logger.trace('0. Prefetch VBOs', this.socket.id, activeBuffers, ticker);

        return driver.fetchData(graph, renderConfig, compress,
                                activeBuffers, lastVersions, activePrograms)
            .do(function (VBOs) {
                logger.trace('1. pre-fetched VBOs for xhr2: ' + sizeInMBOfVBOs(VBOs.compressed) + 'MB', ticker);

                //tell XHR2 sender about it
                if (this.lastCompressedVBOs) {
                    _.extend(this.lastCompressedVBOs, VBOs.compressed);
                } else {
                    this.rememberVBOs(VBOs.compressed);
                }
                this.lastMetadata = {elements: VBOs.elements, bufferByteLengths: VBOs.bufferByteLengths};

                if (saveAtEachStep) {
                    persist.saveVBOs(defaultSnapshotName, VBOs, step);
                }
            }.bind(this))
            .flatMap(function (VBOs) {
                logger.trace('2. Waiting for client to finish previous', this.socket.id, ticker);
                return clientReady
                    .filter(_.identity)
                    .take(1)
                    .do(function () {
                        logger.trace('2b. Client ready, proceed and mark as processing.', this.socket.id, ticker);
                        clientReady.onNext(false);
                    }.bind(this))
                    .map(_.constant(VBOs));
            }.bind(this))
            .flatMap(function (VBOs) {
                logger.trace('3. tell client about availability', this.socket.id, ticker);

                //for each buffer transfer
                var clientAckStartTime;
                var clientElapsed;
                var transferredBuffers = [];
                this.bufferTransferFinisher = function (bufferName) {
                    logger.trace('5a ?. sending a buffer', bufferName, this.socket.id, ticker);
                    transferredBuffers.push(bufferName);
                    //console.log("Length", transferredBuffers.length, requestedBuffers.length);
                    if (transferredBuffers.length === requestedBuffers.length) {
                        logger.trace('5b. started sending all', this.socket.id, ticker);
                        logger.trace('Socket', '...client ping ' + clientElapsed + 'ms');
                        logger.trace('Socket', '...client asked for all buffers',
                            Date.now() - clientAckStartTime, 'ms');
                    }
                }.bind(this);

                // var emitFnWrapper = Rx.Observable.fromCallback(socket.emit, socket);

                //notify of buffer/texture metadata
                //FIXME make more generic and account in buffer notification status
                var receivedAll = colorTexture.flatMap(function (colorTexture) {
                    logger.trace('4a. unwrapped texture meta', ticker);

                    var textures = {
                        colorMap: _.pick(colorTexture, ['width', 'height', 'bytes'])
                    };

                    //FIXME: should show all active VBOs, not those based on prev req
                    var metadata =
                        _.extend(
                            _.pick(VBOs, ['bufferByteLengths', 'elements']),
                            {
                                textures: textures,
                                versions: {
                                    buffers: VBOs.versions,
                                    textures: {colorMap: 1}
                                },
                                step: step
                            });
                    lastVersions = VBOs.versions;
                    lastTick = VBOs.tick;

                    logger.trace('4b. notifying client of buffer metadata', metadata, ticker);
                    //performance monitor here?
                    // profiling.trace('===Sending VBO Update===');

                    //var emitter = socket.emit('vbo_update', metadata, function (time) {
                    //return time;
                    //});
                    //var observableCallback = Rx.Observable.fromNodeCallback(emitter);
                    //return observableCallback;
                    return Rx.Observable.fromCallback(this.socket.emit.bind(this.socket))('vbo_update', metadata);
                    //return emitFnWrapper('vbo_update', metadata);

                }.bind(this)).do(
                    function (clientElapsedMsg) {
                        logger.trace('6. client all received', this.socket.id, ticker);
                        clientElapsed = clientElapsedMsg;
                        clientAckStartTime = Date.now();
                    }.bind(this));

                return receivedAll;
            }.bind(this))
            .flatMap(function () {
                logger.trace('7. Wait for next animation step, updateVboSubject, or if we are behind on ticks', this.socket.id, ticker);

                var filteredUpdateVbo = this.updateVboSubject.filter(function (data) {
                    return data;
                });

                var behindOnTicks = graphObservable.take(1).filter(function (graph) {
                    return graph.simulator.versions.tick > lastTick;
                });

                return Rx.Observable.merge(this.ticksMulti, filteredUpdateVbo, behindOnTicks)
                    .take(1)
                    .do(function (/*data*/) {
                        // Mark that we don't need to send VBOs independently of ticks anymore.
                        this.updateVboSubject.onNext(false);
                    }.bind(this))
                    .do(function () { logger.trace('8. next ready!', this.socket.id, ticker); }.bind(this));
            }.bind(this))
            .map(_.constant(graph));
    }.bind(this))
    .subscribe(function () {
            logger.trace('9. LOOP ITERATED', this.socket.id);
        }.bind(this),
        log.makeRxErrorHandler(logger, 'Main loop failure'));
};


VizServer.prototype.dispose = function () {
    logger.info('disconnecting', this.socket.id);
    delete this.lastCompressedVBOs;
    delete this.bufferTransferFinisher;
    delete this.cachedVBOs[this.socket.id];
    this.isActive = false;
};


if (require.main === module) {

    var url     = require('url');

    var express = require('express');
    var proxy   = require('express-http-proxy');

    var app     = express();
    var http    = require('http').Server(app);
    var io      = require('socket.io')(http, {path: '/worker/3000/socket.io'});

    // Tell Express to trust reverse-proxy connections from localhost, linklocal, and private IP ranges.
    // This allows Express to expose the client's real IP and protocol, not the proxy's.
    app.set('trust proxy', ['loopback', 'linklocal', 'uniquelocal']);

    // debug('Config set to %j', config); //Only want config to print once, which happens when logger is initialized

    var nocache = function (req, res, next) {
        res.header('Cache-Control', 'private, no-cache, no-store, must-revalidate');
        res.header('Expires', '-1');
        res.header('Pragma', 'no-cache');
        next();
    };
    app.use(nocache);

    var allowCrossOrigin = function  (req, res, next) {
        res.header('Access-Control-Allow-Origin', '*');
        res.header('Access-Control-Allow-Headers', 'X-Requested-With,Content-Type,Authorization');
        res.header('Access-Control-Allow-Methods', 'GET,PUT,PATCH,POST,DELETE');
        next();
    };
    app.use(allowCrossOrigin);

    //Static assets
    app.get('*/StreamGL.js', function(req, res) {
        res.sendFile(require.resolve('StreamGL/dist/StreamGL.js'));
    });
    app.get('*/StreamGL.map', function(req, res) {
        res.sendFile(require.resolve('StreamGL/dist/StreamGL.map'));
    });
    app.use('/graph', function (req, res, next) {
        return express.static(path.resolve(__dirname, 'assets'))(req, res, next);
    });

    //Dyn routing
    app.get('/vizaddr/graph', function(req, res) {
        res.json({
            'hostname': config.HTTP_LISTEN_ADDRESS,
            'port': config.HTTP_LISTEN_PORT,
            'timestamp': Date.now()
        });
    });

    // Both keyed by socket ID:
    var servers = {};
    var cachedVBOs = {};

    io.on('connection', function (socket) {
        servers[socket.id] = new VizServer(app, socket, cachedVBOs);
    });

    logger.info('Binding', config.HTTP_LISTEN_ADDRESS, config.HTTP_LISTEN_PORT);
    var listen = Rx.Observable.fromNodeCallback(
            http.listen.bind(http, config.HTTP_LISTEN_PORT, config.HTTP_LISTEN_ADDRESS))();

    listen.do(function () {

        //proxy worker requests
        var from = '/worker/' + config.HTTP_LISTEN_PORT + '/';
        var to = 'http://localhost:' + config.HTTP_LISTEN_PORT;
        logger.info('setting up proxy', from, '->', to);
        app.use(from, proxy(to, {
            forwardPath: function(req/*, res*/) {
                return url.parse(req.url).path.replace(new RegExp('worker/' + config.HTTP_LISTEN_PORT + '/'),'/');
            }
        }));



    }).subscribe(
        function () { logger.info('\nViz worker listening...'); },
        log.makeRxErrorHandler(logger, 'server-viz main')
    );

}


module.exports = VizServer;
