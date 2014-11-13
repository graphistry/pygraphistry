#!/usr/bin/env node
'use strict';

var debug            = require('debug')('StreamGL:master_server');
var config           = require('./config')();
var STATIC_FILE_PATH = require('nodecl').staticFilePath();
var HORIZON_PATH     = require('horizon-viz').staticFilePath() + '/assets/';

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

app.get('/vizaddr/horizon', function(req, res) {
    debug("Assigning client '%s' to viz server on %s, port %d", req.ip, VIZ_SERVER_HOST, VIZ_SERVER_PORT);
    res.json({'hostname': VIZ_SERVER_HOST, 'port': VIZ_SERVER_PORT});
});


// Serve the StreamGL client library
app.get('*/StreamGL.js', function(req, res) {
    res.sendFile(require.resolve('StreamGL/dist/StreamGL.js'));
});
app.get('*/StreamGL.map', function(req, res) {
    res.sendFile(require.resolve('StreamGL/dist/StreamGL.map'));
});

// Serve the horizon demo (/horizon/index.html)
app.use('/horizon', express.static(HORIZON_PATH));

// Serve general static assets (and /sc.html)
app.use(express.static(STATIC_FILE_PATH));


// Default '/' path redirects to graph demo
app.get('/', function(req, res) {
    debug('redirecting')
    res.redirect('/graph.html' + (req.query.debug !== undefined ? '?debug' : ''));
});


// Start listening for HTTP connections
try {
    http.listen(config.LISTEN_PORT, config.LISTEN_ADDRESS, function() {
        console.log('\n[server.js] Server listening on %s:%d', config.LISTEN_ADDRESS, config.LISTEN_PORT);
    });
} catch(e) {
    console.error("[server.js] Fatal error: could not start server on address %s, port %s. Exiting...", config.LISTEN_ADDRESS, config.LISTEN_PORT);
    process.exit(1);
}
