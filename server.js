#!/usr/bin/env node
"use strict";


var Rx = require("rx");
var _ = require("underscore");
var debug = require("debug")("StreamGL:server");

var driver = require("./js/node-driver.js");

var express = require("express");
var app = express();
var http = require("http").Server(app);
/* global -io */ //Set jshint to ignore `predef:"io"` in .jshintrc so we can manually define io here
var io = require("socket.io")(http, {transports: ["websocket"]});


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

app.use(nocache, express.static("/Users/mtorok/Documents/repositories/superconductor2/nodecl/GPUStreaming/"));

// Use the first argument to this script on the command line, if it exists, as the listen port.
var httpPort = process.argv.length > 2 ? process.argv[2] : 10000;
http.listen(httpPort, "localhost", function() {
    console.log("\nServer listening on localhost:" + httpPort);
});


var animStep = driver.create();

io.on("connection", function(socket) {
    debug("Client connected");

    var emitFnWrapper = Rx.Observable.fromCallback(socket.emit, socket);
    var acknowledged = new Rx.BehaviorSubject(0);

    animStep
        .sample(acknowledged)
        .merge(animStep.take(1))  // Ensure we fire one event to kick off the loop
        .flatMap(function(graph) {
            // TODO: Proactively fetch the graph as soon as we've sent the last one, or the data
            // gets stale, and use this data when sending to the client
            return driver.fetchData(graph);
        })
        .subscribe(
            function(vbos) {
                debug("Socket", "Emitting VBOs: " + vboSizeMB(vbos.buffers) + "MB");
                emitFnWrapper("vbo_update", vbos).subscribe(
                    function(clientElapsed) {
                        debug("Socket", "...client acknowledged with time " + clientElapsed + "ms");
                        acknowledged.onNext(clientElapsed);
                    },
                    function(err) { console.err("Error receiving handshake from client:", err); }
                );
            },
            function(err) { console.error("Error sending VBO update:", err); }
        );

});
