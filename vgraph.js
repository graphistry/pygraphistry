var fs = require('fs');
var zlib = require('zlib');
var path = require('path');

var Q = require('q');
var _ = require('underscore');
var pb = require('protobufjs');
var sprintf = require('sprintf-js').sprintf;

var Log         = require('common/logger.js');
var logger      = Log.createLogger('etlworker:vgraph');

var protoFile = path.resolve(__dirname, '../graph-viz/js/libs/graph_vector.proto');
var builder = pb.loadProtoFile(protoFile);
if (builder === null) {
    logger.die(new Error(), 'error: could not build proto');
}
var pb_root = builder.build();

var defaults = {
    'double': NaN,
    'integer': 0,
    'string': 'n/a',
};

// String * String -> Vector
function makeVector(name, type, target) {
    var vector;

    if (type === 'double') {
        vector = new pb_root.VectorGraph.DoubleAttributeVector();
        vector.dest = 'double_vectors';
        vector.transform = parseFloat;
    } else if (type === 'integer') {
        vector = new pb_root.VectorGraph.Int32AttributeVector();
        vector.dest = 'int32_vectors';
        vector.transform = function (x) {
            return parseInt(x) || 0
        };
    } else {
        vector = new pb_root.VectorGraph.StringAttributeVector();
        vector.dest = 'string_vectors';
        vector.transform = function (x) {
            return String(x).trim();
        };
    }

    vector.default = defaults[type];
    vector.name = name;
    vector.target = target;
    vector.values = [];
    vector.map = {};
    return vector;
}

// JSON -> {String -> Vector}
function getAttributeVectors(header, target) {
    var map = _.map(header, function (info, key) {
        if (info.type === 'empty') {
            logger.info('Skipping attribute', key, 'because it has no data.');
            return [];
        }
        var vec = makeVector(key, info.type, target);
        return [key, vec];
    });

    return _.object(_.filter(map, function (x) {return x.length > 0;}));
}

function defined(value) {
    return value !== undefined && value !== null &&
        value !== '' && value !== 'n/a' &&
        !(typeof value === 'number' && isNaN(value));
}

function inferType(samples) {
    if (samples.length == 0)
        return 'empty';
    if (_.all(samples, function (val) { return !isNaN(val); })) {
        if (_.all(samples, function (val) { return val === +val && val === (val|0); })) {
            return 'integer'
        } else {
            return 'double';
        }
    } else {
        return 'string';
    }
}

function getHeader(table) {
    var res = {};

    var total = 0;

    _.each(table, function (row) {
        _.each(_.keys(row), function (key) {

            var data = res[key] || {count: 0, samples: [], type: undefined};
            var val = row[key];
            if (defined(val)) {
                data.count++;
                if (data.samples.length < 100) {
                    data.samples.push(val);
                }
            }
            res[key] = data;
        });
        total++;
    })

    return _.object(_.map(res, function (data, name) {
        data.freq = data.count / total;
        data.type = inferType(data.samples);
        return [name, data];
    }));
}

// Simple (and dumb) conversion of JSON edge lists to VGraph
// JSON * String * String * String -> VGraph
function fromEdgeList(elist, nlabels, srcField, dstField, idField,  name) {
    var node2Idx = {};
    var idx2Node = {};
    var nodeCount = 0;
    var edges = [];
    // For detecting duplicate edges.
    var edgeMap = {}

    var addNode = function (node) {
        if (!(node in node2Idx)) {
            idx2Node[nodeCount] = node;
            node2Idx[node] = nodeCount;
            nodeCount++;
        }
    };

    var warnsLeftDuplicated = 100;
    var warnsLeftBi = 100;
    var warnsLeftNull = 100;
    var warnsLeftSelf = 100;
    // 'a * 'a -> bool
    // return true if dupe
    var isBadEdge = function (src, dst) {

        var dsts = edgeMap[src] || {};
        if (dst in dsts) {
            if (warnsLeftDuplicated-- > 0) {
                logger.info('Edge %s -> %s is duplicated', src, dst);
            }
            return true;
        }

        if (src === undefined || dst === undefined || src === null || dst === null) {
            if (warnsLeftNull-- > 0) {
                logger.info('Edge %s <-> %s has null field', src, dst);
            }
            return true;
        }

        if (src === dst) {
            if (warnsLeftSelf-- > 0) {
                logger.info('Edge %s <-> %s is a self-edge', src, dst);
            }
            return true;
        }

        return false;
    };

    //return whether added
    // -> bool
    function addEdge(node0, node1, entry) {

        var e = new pb_root.VectorGraph.Edge();
        e.src = node2Idx[node0];
        e.dst = node2Idx[node1];
        edges.push(e);

        var dsts = edgeMap[node0] || {};
        dsts[node1] = true;
        edgeMap[node0] = dsts;

        return true;
    }

    function addAttributes(vectors, entry) {
        _.each(vectors, function (vector, name) {
            if (name in entry && entry[name] !== null && entry[name] !== undefined) {
                vector.values.push(vector.transform(entry[name]));
            } else {
                vector.values.push(vector.default);
            }
        });
    }

    logger.debug('Infering schema...');

    //TODO: log this in a better way, i.e. without saying "Edge Table" before logging the edge table itself
    var eheader = getHeader(elist);
    logger.info('Edge Table');
    _.each(eheader, function (data, key) {
        logger.info(sprintf('%36s: %3d%% filled    %s', key, Math.floor(data.freq * 100).toFixed(0), data.type));
    });
    var nheader = getHeader(nlabels);
    logger.info('Node Table');
    _.each(nheader, function (data, key) {
        logger.info(sprintf('%36s: %3d%% filled    %s', key, Math.floor(data.freq * 100).toFixed(0), data.type));
    });

    if (!(srcField in eheader)) {
        logger.warn('Edges have no srcField' , srcField, 'header', eheader);
        return undefined;
    }
    if (!(dstField in eheader)) {
        logger.warn('Edges have no dstField' , dstField);
        return undefined;
    }
    if (nlabels.length > 0 && !(idField in nheader)) {
        logger.warn('Nodes have no idField' , idField);
        return undefined;
    }
    var evectors = getAttributeVectors(eheader, pb_root.VectorGraph.AttributeTarget.EDGE);
    var nvectors = getAttributeVectors(nheader, pb_root.VectorGraph.AttributeTarget.VERTEX);

    logger.debug('Loading', elist.length, 'edges...');
    _.each(elist, function (entry) {
        var node0 = entry[srcField];
        var node1 = entry[dstField];
        addNode(node0);
        addNode(node1);
        if (!isBadEdge(node0, node1)) {
            //must happen after addNode
            addEdge(node0, node1);

            addAttributes(evectors, entry);
        }
    });

    logger.debug('Loading', nlabels.length, 'labels for', nodeCount, 'nodes');
    if (nodeCount > nlabels.length) {
        logger.info('There are', nodeCount - nlabels.length, 'labels missing');
    }

    var sortedLabels = new Array(nodeCount);
    var warnsLeftLabel = 100;
    for (var i = 0; i < nlabels.length; i++) {
        var label = nlabels[i];
        var nodeId = label[idField];
        if (nodeId in node2Idx) {
            var labelIdx = node2Idx[nodeId];
            sortedLabels[labelIdx] = label;
        } else {
            if (warnsLeftLabel-- > 0) {
                logger.info(sprintf('Skipping label #%6d (nodeId: %10s) which has no matching node. (ID field: %s, label: %s)', i, nodeId, idField, JSON.stringify(label)));
            }
        }
    }

    _.each(sortedLabels, function (entry) {
        addAttributes(nvectors, entry || {});
    });

    logger.debug('Encoding protobuf...');
    var vg = new pb_root.VectorGraph();
    vg.version = 0;
    vg.name = name;
    vg.type = pb_root.VectorGraph.GraphType.DIRECTED;
    vg.nvertices = nodeCount;
    vg.nedges = edges.length;
    vg.edges = edges;

    _.each(_.omit(evectors, srcField, dstField), function (vector) {
        vg[vector.dest].push(vector);
    });

    _.each(_.omit(nvectors, '_mkv_child', '_timediff'), function (vector) {
        vg[vector.dest].push(vector);
    });

    return vg;
}

function decodeVGraph(buffer) {
    return pb_root.VectorGraph.decode(buffer);
}

module.exports = {
    fromEdgeList: fromEdgeList,
    decodeVGraph: decodeVGraph,
};
