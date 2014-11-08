#!/usr/bin/env node
'use strict';

var debug            = require('debug')('StreamGL:master_server');
var config           = require('./config')();
var STATIC_FILE_PATH = require('nodecl').staticFilePath();

debug("Config set to %j", config);

// FIXME: Get real viz server public IP/DNS name from DB
var VIZ_SERVER_HOST = 'localhost';
// FIXME: Get real viz server port from DB
var VIZ_SERVER_PORT = config.LISTEN_PORT + 1;


var express = require('express'),
    app = express(),
    http = require('http').Server(app);


app.get('/vizaddr/graph', function(req, res) {
    debug("Assigning client '%s' to viz server on %s, port %d", req.ip, VIZ_SERVER_HOST, VIZ_SERVER_PORT);
    res.json({'hostname': VIZ_SERVER_HOST, 'port': VIZ_SERVER_PORT});
});


// Serve the socket.io client library
app.get('*/socket.io.js', function(req, res) {
    res.sendFile(require.resolve('socket.io-client/socket.io.js'));
});

app.get('*/StreamGL.js', function(req, res) {
    res.sendFile(require.resolve('StreamGL/dist/StreamGL.js'));
});
app.get('*/StreamGL.map', function(req, res) {
    res.sendFile(require.resolve('StreamGL/dist/StreamGL.map'));
});

app.use(express.static(STATIC_FILE_PATH));

// Default '/' path redirects to graph demo
app.get('/', function(req, res) {
    res.redirect('/graph.html' + (req.query.debug !== undefined ? '?debug' : ''));
});

http.listen(config.LISTEN_PORT, config.LISTEN_ADDRESS, function() {
    console.log('\nServer listening on %s:%d', config.LISTEN_ADDRESS, config.LISTEN_PORT);
});
