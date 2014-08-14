#!/usr/bin/env node

"use strict";

var Rx = require("rx");
var _ = require("underscore");
var debug = require("debug")("StreamGL:server");

var driver = require("./js/node-driver.js");

var express = require("express");
var app = express();
var http = require("http").Server(app);
var io = require("socket.io")(http); // , {transports: ["websocket"]}


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


var vboUpdated = driver.create();

io.on("connection", function(socket) {
    debug("Client connected");

    var emitObs = Rx.Observable.fromCallback(socket.emit, socket);
    var acknowledged = new Rx.BehaviorSubject(true);

    vboUpdated
        .sample(acknowledged)
        .merge(vboUpdated.take(1))  // Ensure we fire one event to kick off the loop
        .subscribe(
            function(vbos) {
                var vboSizeBytes = _.reduce(_.values(vbos.buffers), function(sum, v) {
                        return sum + v.byteLength;
                    }, 0 );
                var vboSizeMB = Math.round((Math.round(vboSizeBytes / 1024) / 1024) * 100) / 100;
                debug("Socket", "Emitting VBOs: " + vboSizeMB + "MB");

                emitObs("vbo_update", vbos)
                    .subscribe(function(clientElapsed) {
                        debug("Socket", "...client acknowledged. Reported performance: " +
                            clientElapsed + "ms");
                        acknowledged.onNext();
                    },
                    function(err) { console.err("Error receiving handshake from client:", err); });
            },
            function(err) {
                console.error("Error sending VBO update:", err);
            }
        )
});
