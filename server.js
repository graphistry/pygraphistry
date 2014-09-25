#!/usr/bin/env node
"use strict";

var NODE_CL_PATH = "/Users/lmeyerov/Desktop/Superconductor2/nodecl/",
    GPU_STREAMING_PATH = NODE_CL_PATH + "GPUStreaming/",
    STREAMGL_PATH = GPU_STREAMING_PATH + "StreamGL/src/";


var Rx          = require("rx"),
    _           = require("underscore"),
    debug       = require("debug")("StreamGL:server");

var driver      = require("./js/node-driver.js"),
    compress    = require(NODE_CL_PATH + "/compress/compress.js"),
    proxyUtils  = require(STREAMGL_PATH + 'proxyutils.js'),
    renderer    = require(STREAMGL_PATH + 'renderer.js');


//FIXME CHEAP HACK TO WORK AROUND CONFIG FILE INCLUDE PATH
var cwd = process.cwd();
process.chdir(GPU_STREAMING_PATH + 'StreamGL');
var renderConfig = require(STREAMGL_PATH + 'renderer.config.graph.js');
process.chdir(cwd);


var express = require("express"),
    app = express(),
    http = require("http").Server(app),
/* global -io */ //Set jshint to ignore `predef:"io"` in .jshintrc so we can manually define io here
    io = require("socket.io")(http, {transports: ["websocket"]}),
    connect = require('connect');

/** Given an Object with buffers as values, returns the sum size in megabytes of all buffers */
function vboSizeMB(vbos) {
    var vboSizeBytes = _.reduce(_.values(vbos.buffers), function(sum, v) {
            return sum + v.byteLength;
        }, 0);
    return Math.round((Math.round(vboSizeBytes / 1024) / 1024) * 100) / 100;
}



function nocache(req, res, next) {
    res.header("Cache-Control", "private, no-cache, no-store, must-revalidate");
    res.header("Expires", "-1");
    res.header("Pragma", "no-cache");
    next();
}

app.use(nocache, express.static(GPU_STREAMING_PATH));

// Use the first argument to this script on the command line, if it exists, as the listen port.
var httpPort = process.argv.length > 2 ? process.argv[2] : 10000;
http.listen(httpPort, "localhost", function() {
    console.log("\nServer listening on localhost:" + httpPort);
});


//Serve most recent compressed binary buffers
var lastCompressedVbos = {};
var finishBufferTransfer = function () {};
var server = connect()
    .use(function (req, res, next) {
        var bufferName = req.url.split('=')[1];
        try {
            res.writeHead(200, {
                'Content-Encoding': 'gzip',
                'Content-Type': 'text/javascript',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'X-Requested-With,Content-Type,Authorization',
                'Access-Control-Allow-Methods': 'GET,PUT,PATCH,POST,DELETE'
            });
            res.end(lastCompressedVbos[bufferName]);
        } catch (e) {
            console.error('bad request', e, e.stack);
        }
        finishBufferTransfer(bufferName);
    })
    .listen(proxyUtils.BINARY_PORT);




var animStep = driver.create();

io.on("connection", function(socket) {
    debug("Client connected");

    var emitFnWrapper = Rx.Observable.fromCallback(socket.emit, socket);
    var acknowledged = new Rx.BehaviorSubject(0);

    var activeBuffers = renderer.getServerBufferNames(renderConfig);
    var activePrograms = renderConfig.scene.render;

    var lastGraph = null;
    socket.on('received_buffers', function (time) {
        debug("Client end-to-end time", time);
        acknowledged.onNext(lastGraph);
    });

    animStep
        .sample(acknowledged)
        .merge(animStep.take(1))  // Ensure we fire one event to kick off the loop
        .flatMap(function(graph) {
            // TODO: Proactively fetch the graph as soon as we've sent the last one, or the data
            // gets stale, and use this data when sending to the client
            lastGraph = graph;
            return driver.fetchData(graph, compress, activeBuffers, activePrograms);
        })
        .subscribe(
            function(vbos) {

                debug("Socket", "Emitting VBOs: " + vboSizeMB(vbos.compressed) + "MB");

                lastCompressedVbos = vbos.compressed;

                emitFnWrapper("vbo_update", _.pick(vbos, ['bufferByteLengths', 'elements']))
                    .subscribe(
                        function(clientElapsed) {

                            var clientAckStartTime = Date.now();

                            var transferredBuffers = [];
                            finishBufferTransfer = function (bufferName) {
                                transferredBuffers.push(bufferName);
                                if (transferredBuffers.length == activeBuffers.length) {
                                    debug("Socket", "...client ping " + clientElapsed + "ms");
                                    debug("Socket", "...client asked for all buffers",
                                        Date.now() - clientAckStartTime, 'ms');
                                }
                            };

                        },
                        function(err) { console.err("Error receiving handshake from client:", err); }
                    );
            },
            function(err) { console.error("Error sending VBO update:", err); }
        );

});
