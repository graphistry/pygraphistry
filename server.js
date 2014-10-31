#!/usr/bin/env node
'use strict';

var debug       = require('debug')('StreamGL:master_server');
var config      = require('./config')();
var StreamGL    = require('StreamGL');  // For STATIC_HTTP_PATH

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
app.use(express.static(StreamGL.STATIC_HTTP_PATH));

// Default '/' path redirects to graph demo
app.get('/', function(req, res) {
    res.redirect('/graph.html' + (req.query.debug !== undefined ? '?debug' : ''));
});

http.listen(config.LISTEN_PORT, config.LISTEN_ADDRESS, function() {
    console.log('\nServer listening on %s:%d', config.LISTEN_ADDRESS, config.LISTEN_PORT);
});
