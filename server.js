#!/usr/bin/env node
"use strict";

var config      = require('./config')(process.argv.length > 2 ? JSON.parse(process.argv[2]) : {});

var Rx          = require("rx"),
    _           = require("underscore"),
    debug       = require("debug")("StreamGL:server");

var driver      = require("./js/node-driver.js"),
    compress    = require(config.NODE_CL_PATH + "/compress/compress.js"),
    proxyUtils  = require(config.STREAMGL_PATH + 'proxyutils.js'),
    renderer    = require(config.STREAMGL_PATH + 'renderer.js');


//FIXME CHEAP HACK TO WORK AROUND CONFIG FILE INCLUDE PATH
var cwd = process.cwd();
process.chdir(config.GPU_STREAMING_PATH + 'StreamGL');
var renderConfig = require(config.STREAMGL_PATH + 'renderer.config.graph.js');
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

app.use(nocache, express.static(config.GPU_STREAMING_PATH));

// If an argument is supplied to this script, use it as the listening address:port
var listenAddress = config.DEFAULT_LISTEN_ADDRESS;
var listenPort = config.DEFAULT_LISTEN_PORT;

http.listen(listenPort, listenAddress, function() {
    console.log("\nServer listening on %s:%d", listenAddress, listenPort);
});


//Serve most recent compressed binary buffers
//FIXME this seems broken in case of multiuser
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

//make available to all clients
var graph = new Rx.ReplaySubject(1);
animStep.ticks.take(1).subscribe(graph);

io.on("connection", function(socket) {
    debug("Client connected");


    var activeBuffers = renderer.getServerBufferNames(renderConfig);
    var activePrograms = renderConfig.scene.render;

    socket.on('graph_settings', function (payload) {
        debug('new settings', payload);
        animStep.proxy(payload);
    });


    // ============= EVENT LOOP

    //starts true, set to false whenever transfer starts, true again when ack'd
    var clientReady = new Rx.ReplaySubject(1);
    clientReady.onNext(true);
    socket.on('received_buffers', function (time) {
        debug("Client end-to-end time", time);
        clientReady.onNext(true);
    });

    clientReady.subscribe(
        debug.bind('CLIENT STATUS'));

    debug('SETTING UP CLIENT EVENT LOOP');
    graph.expand(function (graph) {

        debug('1. Prefetch VBOs')
        return driver.fetchData(graph, compress, activeBuffers, activePrograms)
            .do(function (vbos) {
                debug("prefetched VBOs for xhr2: " + vboSizeMB(vbos.compressed) + "MB");
                //tell XHR2 sender about it
                lastCompressedVbos = vbos.compressed;
            })
            .flatMap(function (vbos) {
                debug('2. Waiting for client to finish previous');
                return clientReady
                    .filter(_.identity)
                    .take(1)
                    .do(function () {
                        debug('2b. Client ready, proceed and mark as processing.')
                        clientReady.onNext(false);
                    })
                    .map(_.constant(vbos))
            })
            .flatMap(function (vbos) {
                debug('3. tell client about availablity');

                //for each buffer transfer
                var sendingAllBuffers = new Rx.Subject();
                var clientAckStartTime;
                var clientElapsed;
                var transferredBuffers = [];
                finishBufferTransfer = function (bufferName) {
                    debug('3a ?. sending a buffer', bufferName)
                    transferredBuffers.push(bufferName);
                    if (transferredBuffers.length == activeBuffers.length) {
                        debug('3b. started sending all');
                        debug("Socket", "...client ping " + clientElapsed + "ms");
                        debug("Socket", "...client asked for all buffers",
                            Date.now() - clientAckStartTime, 'ms');
                        sendingAllBuffers.onNext();
                    }
                };

                var emitFnWrapper = Rx.Observable.fromCallback(socket.emit, socket);
                var sending = emitFnWrapper("vbo_update",
                    _.pick(vbos, ['bufferByteLengths', 'elements']));

                sending.subscribe(function (clientElapsedMsg) {
                        debug('3d ?. client all received')
                        clientElapsed = clientElapsedMsg;
                        clientAckStartTime = Date.now();
                    });

                return sendingAllBuffers
                    .take(1)
                    .do(debug.bind('3c. All in transit'));
            })
            .flatMap(function () {
                debug('4. Wait for next anim step');
                return animStep.ticks
                    .take(1)
                    .do(function () { debug('4b. next ready!'); });
            })
            .map(_.constant(graph));
    })
    .subscribe(function () { debug('LOOP ITERATED.') });

});
