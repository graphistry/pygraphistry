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

VizServer.prototype.resetState = function (dataset) {
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
    this.animationStep = driver.create(dataset);
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
        graph.simulator.selectNodes(query.sel).then(function (nodeIndices) {
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
            cb({success: false, error: 'aggregate error'});
            log.makeRxErrorHandler(logger, 'aggregate handler')(err);
        }
    );
};

// TODO Extract a graph method and manage graph contexts by filter data operation.
VizServer.prototype.filterGraphByMaskList = function (graph, maskList, errors, viewConfig, pointLimit, cb) {
    var filters = viewConfig.filters;
    var masks = graph.dataframe.composeMasks(maskList, pointLimit);

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
                var response = {success: true, filters: filters, sets: sets};
                if (errors) {
                    response.errors = errors;
                }
                cb(response);
            }.bind(this)).done(_.identity, function (err) {
                log.makeQErrorHandler(logger, 'dataframe filter')(err);
                errors.push(err);
                var response = {success: false, errors: errors, filters: filters};
                cb(response);
            });
    } catch (err) {
        log.makeQErrorHandler(logger, 'dataframe filter')(err);
        errors.push(err);
        var response = {success: false, errors: errors, filters: filters};
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
            data = [function () {return graph.dataframe.aggregate(graph.simulator, indices[query.type], query.attributes, query.binning, query.mode, query.type);}];
        } else {
            var types = ['point', 'edge'];
            data = _.map(types, function (type) {
                var filteredAttributes = _.filter(query.attributes, function (attr) {
                    return (attr.type === type);
                });
                var attributeNames = _.pluck(filteredAttributes, 'name');
                return function () {
                    return graph.dataframe.aggregate(graph.simulator, indices[type], attributeNames, query.binning, query.mode, type);
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

function presentVizSet(vizSet) {
    if (vizSet.masks === undefined) { return vizSet; }
    var maskResponseLimit = 3e3;
    var masksTooLarge = vizSet.masks.numPoints() > maskResponseLimit ||
        vizSet.masks.numEdges() > maskResponseLimit;
    var response = masksTooLarge ? _.omit(vizSet, ['masks']) : _.clone(vizSet);
    response.sizes = {point: vizSet.masks.numPoints(), edge: vizSet.masks.numEdges()};
    // Do NOT serialize the dataframe.
    if (response.masks && response.masks.dataframe !== undefined) {
        response.masks = _.omit(response.masks, 'dataframe');
    }
    return response;
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
                // vizSet.masks = ??
                break;
        }
    });
    return _.map(sets, presentVizSet);
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

            this.resetState(dataset);
            renderConfigDeferred.resolve(rConf.scenes[metadata.scene]);
        }.bind(this)).fail(log.makeQErrorHandler(logger, 'resetting state'));
    }.bind(this)).subscribe(_.identity, log.makeRxErrorHandler(logger, 'Get render config'));

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
            cb({success: false, error: 'Render config read error'});
            log.makeQErrorHandler(logger, 'sending render_config')(err);
            cb({success: false, error: 'Render config read error'});
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
            cb({success: false, error: 'Render config update error'});
            log.makeQErrorHandler(logger, 'updating render_config')(err);
        });
    }.bind(this));

    /**
     * @typedef {Object} SetSpecification
     * @property {String} sourceType one of selection,dataframe,filtered
     * @property {Object} sel rectangle/etc selection gesture.
     * @property {Number[]} point_ids list of point IDs.
     */

    this.socket.on('create_set', function (specification, name, cb) {
        Rx.Observable.combineLatest(this.graph, this.viewConfig, function (graph, viewConfig) {
            var qNodeSelection;
            var sourceType = specification.sourceType;
            if (sourceType === 'selection' || sourceType === undefined) {
                if (specification.sel !== undefined) {
                    var selection = specification.sel;
                    qNodeSelection = graph.simulator.selectNodes(selection);
                } else if (_.isArray(specification.point_ids)) {
                    qNodeSelection = Q(specification.point_ids);
                } else {
                    throw Error('Selection not specified for creating a Set');
                }
                qNodeSelection = qNodeSelection.then(function (pointIndexes) {
                    var edgeIndexes = graph.simulator.connectedEdges(pointIndexes);
                    return new DataframeMask(graph.dataframe, pointIndexes, edgeIndexes);
                });
            } else if (sourceType === 'dataframe') {
                qNodeSelection = Q(graph.dataframe.fullDataframeMask());
            } else if (sourceType === 'filtered') {
                qNodeSelection = Q(graph.dataframe.lastMasks);
            } else {
                throw Error('Unrecognized special type for creating a Set: ' + sourceType);
            }
            qNodeSelection.then(function (dataframeMask) {
                var newSet = {
                    id: new TransactionalIdentifier().toString(),
                    name: name,
                    masks: dataframeMask,
                    sizes: {point: dataframeMask.numPoints(), edge: dataframeMask.numEdges()}
                };
                viewConfig.sets.push(newSet);
                this.dataframe.masksForVizSets[newSet.id] = dataframeMask;
                cb({success: true, set: presentVizSet(newSet)});
            }).fail(log.makeQErrorHandler(logger, 'pin_selection_as_set'));
        }).take(1).subscribe(_.identity,
            function (err) {
                logger.error(err, 'Error creating set from selection');
                cb({success: false, error: 'Server error when saving the selection as a Set'});
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
                cb({success: false, error: 'Server error when retrieving all Set definitions'});
            });
    }.bind(this));

    /**
     * This handles creates (set given with no id), updates (id and set given), and deletes (id with no set).
     */
    this.socket.on('update_set', function (id, updatedVizSet, cb) {
        this.viewConfig.take(1).do(function (viewConfig) {
            if (_.contains(specialSetKeys, id)) {
                throw Error('Cannot update the special Sets');
            }
             var matchingSetIndex = _.findIndex(viewConfig.sets, function (vizSet) { return vizSet.id === id; });
            if (matchingSetIndex === -1) {
                // Auto-create:
                if (updatedVizSet === undefined) {
                    updatedVizSet = {};
                }
                // Auto-create an ID:
                if (updatedVizSet.id === undefined) {
                    updatedVizSet.id = (id || new TransactionalIdentifier()).toString();
                }
                viewConfig.sets.push(updatedVizSet);
            } else {
                // Delete as un-define:
                if (updatedVizSet === undefined) {
                    viewConfig.splice(matchingSetIndex, 1);
                } else {
                    if (updatedVizSet.id === undefined) {
                        updatedVizSet.id = id;
                    }
                    // TODO: smart merge
                    viewConfig.sets[matchingSetIndex] = updatedVizSet;
                }
            }
            cb({success: true, set: presentVizSet(updatedVizSet)});
        }).subscribe(_.identity,
            function (err) {
                logger.error(err, 'Error sending update_set');
                cb({success: false, error: 'Server error when updating a Set'});
                throw err;
            });
    }.bind(this));

    this.socket.on('get_filters', function (cb) {
        logger.trace('sending current filters to client');
        this.viewConfig.take(1).do(function (viewConfig) {
            cb({success: true, filters: viewConfig.filters});
        }).subscribe(
            _.identity, log.makeRxErrorHandler(logger, 'get_filters handler'));
    }.bind(this));

    this.socket.on('update_filters', function (newValues, cb) {
        logger.trace('updating filters from client values');
        // Maybe direct assignment isn't safe, but it'll do for now.
        this.viewConfig.take(1).do(function (viewConfig) {
            if (!_.isEqual(newValues, viewConfig.filters)) {
                viewConfig.filters = newValues;
                this.viewConfig.onNext(viewConfig);
            }
            logger.info('updated filters', viewConfig.filters);

            this.graph.take(1).do(function (graph) {
                var dataframe = graph.dataframe;
                var maskList = [];
                var errors = [];
                var pointLimit = Infinity;

                _.each(viewConfig.filters, function (filter) {
                    if (filter.enabled === false) {
                        return;
                    }
                    /** @type ClientQuery */
                    var filterQuery = filter.query;
                    var masks;
                    if (filterQuery === undefined) {
                        return;
                    }
                    var ast = filterQuery.ast;
                    if (ast !== undefined &&
                        ast.type === 'Limit' &&
                        ast.value !== undefined) {
                        var generator = new ExpressionCodeGenerator('javascript');
                        pointLimit = generator.evaluateExpressionFree(ast.value);
                        return;
                    }
                    var type = filter.type || filterQuery.type;
                    var attribute = filter.attribute || filterQuery.attribute;
                    var normalization = dataframe.normalizeAttributeName(filterQuery.attribute, type);
                    if (normalization === undefined) {
                        errors.push('Unknown frame element');
                        return;
                    } else {
                        type = normalization.type;
                        attribute = normalization.attribute;
                    }
                    if (type === 'point') {
                        var pointMask = dataframe.getPointAttributeMask(attribute, filterQuery);
                        masks = dataframe.masksFromPoints(pointMask);
                    } else if (type === 'edge') {
                        var edgeMask = dataframe.getEdgeAttributeMask(attribute, filterQuery);
                        masks = dataframe.masksFromEdges(edgeMask);
                    } else {
                        errors.push('Unknown frame element type');
                        return;
                    }
                    // Record the size of the filtered set for UI feedback:
                    filter.maskSizes = {point: masks.numPoints(), edge: masks.numEdges()};
                    maskList.push(masks);
                });

                this.filterGraphByMaskList(graph, maskList, errors, viewConfig, pointLimit, cb);
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
            cb({success: false, error: 'Server error when fetching controls'});
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
            this.resetState(dataset);
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
                cb({success: false, error: 'inspect_header error'});
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
                cb({success: false, error: 'Namespace metadata error'});
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
            cb({success: false, error: 'Namespace metadata update error'});
            log.makeQErrorHandler(logger, 'updating namespace metadata');
        });
    }.bind(this));

    this.socket.on('filter', function (query, cb) {
        logger.info('Got filter', query);
        Rx.Observable.combineLatest(this.viewConfig, this.graph, function (viewConfig, graph) {

            var maskList = [];
            var errors = [];

            var dataframe = graph.dataframe;
            _.each(query, function (data, attribute) {
                var masks;
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
                if (type === 'point') {
                    var pointMask = dataframe.getPointAttributeMask(attribute, data);
                    masks = dataframe.masksFromPoints(pointMask);
                } else if (type === 'edge') {
                    var edgeMask = dataframe.getEdgeAttributeMask(attribute, data);
                    masks = dataframe.masksFromEdges(edgeMask);
                } else {
                    errors.push('Unrecognized type: ' + type);
                    cb({success: false, errors: errors});
                    return;
                }
                maskList.push(masks);
            });
            this.filterGraphByMaskList(graph, maskList, errors, viewConfig, Infinity, cb);
        }.bind(this)).take(1).subscribe(
            _.identity,
            function (err) {
                log.makeRxErrorHandler(logger, 'aggregate handler')(err);
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
                qIndices = graph.simulator.selectNodes(query.sel);
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
                cb({success: false, error: 'aggregate socket error'});
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

    this.socket.on('shortest_path', function (pair) {
        graph.take(1)
            .do(function (graph) {
                graph.simulator.highlightShortestPaths(pair);
                animationStep.interact({play: true, layout: true});
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

    this.socket.on('highlight_points', function (points) {
        graph.take(1)
            .do(function (graph) {

                points.forEach(function (point) {
                    graph.simulator.dataframe.getLocalBuffer('pointColors')[point.index] = point.color;
                    // graph.simulator.buffersLocal.pointColors[point.index] = point.color;
                });
                graph.simulator.tickBuffers(['pointColors']);

                animationStep.interact({play: true, layout: true});
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
                    return cb({success: false, error: rejectedResult});
                });
            }).take(1).subscribe(_.identity, log.makeRxErrorHandler(logger, 'persist_current_workbook'));
    }.bind(this));

    this.socket.on('persist_current_vbo', function(contentKey, cb) {
        graph.take(1)
            .do(function (graph) {
                var cleanContentKey = encodeURIComponent(contentKey);
                persist.publishStaticContents(
                    cleanContentKey, this.lastCompressedVBOs,
                    this.lastMetadata, graph.dataframe, renderConfig).then(function() {
                    cb({success: true, name: cleanContentKey});
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
                cb({success: false, error: 'fork_vgraph error'});
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
                logger.trace('7. Wait for next animation step', this.socket.id, ticker);

                var filteredUpdateVbo = this.updateVboSubject.filter(function (data) {
                    return data;
                });

                return this.ticksMulti.merge(filteredUpdateVbo)
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
