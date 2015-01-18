var Q = require('q');
var _ = require('underscore');
var fs = require('fs');
var debug = require('debug')('graphistry:splunk:vgraph');
var pb = require('protobufjs');
var zlib = require('zlib');
var path = require('path');

var builder = null;
var pb_root = null;
var protoFile = path.resolve(__dirname, '../js/libs/graph_vector.proto');

pb.loadProtoFile(protoFile, function (err, builder_) {
    if (err) {
        debug('error: could not build proto', err, err.stack);
        return;
    } else {
        builder = builder_;
        pb_root = builder.build();
    }
});

function fromEdgeList(elist, name) {
    var node2Idx = {};
    var nodeCount = 0;
    var edges = [];

    function addNode(node) {
        if (!(node in node2Idx)) {
            node2Idx[node] = nodeCount++;
        }
    }

    function addEdge(node0, node1, entry) {
        var e = new pb_root.VectorGraph.Edge();
        e.src = node2Idx[node0];
        e.dst = node2Idx[node1];
        edges.push(e);
    }

    function getAttributeVectors(entry) {
        return _.object(_.map(_.keys(entry), function (key) {
            var vector;
            var val = entry[key];

            if (!isNaN(Number(val))) {
                vector = new pb_root.VectorGraph.DoubleAttributeVector();
                vector.dest = 'double_vectors';
                vector.transform = parseFloat;
            } else {
                vector = new pb_root.VectorGraph.StringAttributeVector();
                vector.dest = 'string_vectors';
                vector.transform = _.identity;
            }

            vector.name = key;
            vector.target = pb_root.VectorGraph.AttributeTarget.EDGE;
            vector.values = [];
            return [key, vector];
        }));
    }

    function addAttributes(entry) {
        _.each(entry, function (val, key) {
            var vector = vectors[key];
            vector.values.push(vector.transform(val));
        });
    }

    var vectors = getAttributeVectors(elist[0]);

    for (var i = 0; i < elist.length; i++) {
        var entry = elist[i];
        var node0 = entry.src_ip + ':0';
        var node1 = entry.dest_ip + ':' + entry.dest_port;
        addNode(node0);
        addNode(node1);
        addEdge(node0, node1);
        // Assumes that all edges have the same attributes.
        addAttributes(entry);
    }

    var vg = new pb_root.VectorGraph();
    vg.version = 0;
    vg.name = name;
    vg.type = pb_root.VectorGraph.GraphType.DIRECTED;
    vg.nvertices = nodeCount;
    vg.nedges = edges.length;
    vg.edges = edges;

    _.each(vectors, function (vector) {
        vg[vector.dest].push(vector);
    })

    debug('VectorGraph', vg);

    return vg;
}

module.exports = {
    fromEdgeList: fromEdgeList
};
