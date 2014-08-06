#!/usr/bin/env node

"use strict";

var Rx = require("rx");
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
    console.log("listening on localhost:" + httpPort);
});

io.on("connection", function(socket) {
    var vboUpdated = driver.create();
    vboUpdated.subscribe(function(vbos) {
        console.debug("VBOs updated");
    })
});
