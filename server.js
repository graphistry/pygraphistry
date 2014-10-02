#!/usr/bin/env node
"use strict";

var NODE_CL_PATH = "/opt/Superconductor2/nodecl/",
    GPU_STREAMING_PATH = NODE_CL_PATH + "GPUStreaming/",
    STREAMGL_PATH = GPU_STREAMING_PATH + "StreamGL/src/";

// Default IP and port the server listens on. Can be overridden by the user by passing an argument
// to this script on the command line of form <IP>:<PORT>. <IP> is either 4 numbers ('192.169.0.1')
// or 'localhost'; <PORT> is a number. Both are optional. If only 1 is supplied, ':' is optional.
var DEFAULT_LISTEN_ADDRESS = 'localhost';
var DEFAULT_LISTEN_PORT = 10000;


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

// If an argument is supplied to this script, use it as the listening address:port
var listenAddress = DEFAULT_LISTEN_ADDRESS;
var listenPort = DEFAULT_LISTEN_PORT;
if(process.argv.length > 2) {
    var addressParts = process.argv[2].match(
        /^(([0-9]{1,3}\.){3}[0-9]{1,3}|localhost)?(:?([0-9]+)?)?$/i);

    var listenAddress = addressParts[1] !== undefined && addressParts[1] !== "" ?
        addressParts[1] : DEFAULT_LISTEN_ADDRESS;
    var listenPort = addressParts[4] !== undefined && addressParts[4] !== "" ?
        parseInt(addressParts[4], 10) : DEFAULT_LISTEN_PORT;
}


http.listen(listenPort, listenAddress, function() {
    console.log("\nServer listening on %s:%d", listenAddress, listenPort);
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

            var retryCount = 0;

            lastGraph = graph;
            return Rx.Observable.return(1)
                .flatMap(function () {
                    retryCount++;
                    if (retryCount > 1) {
                        console.error('retrying', retryCount);
                    }
                    return Rx.Observable.return(1)
                        .delay(retryCount - 1)
                        .flatMap(function () {
                            return driver.fetchData(graph, compress, activeBuffers, activePrograms);
                        });
                })
                .retry(25);
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
            function(err) { console.error("Error sending VBO update:", err, err.stack); }
        );

});
