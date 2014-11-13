#!/usr/bin/env node
'use strict';

var path   = require('path');
var debug  = require('debug')('StreamGL:master_server');
var config = require('./config')();
var mongo  = require('mongodb');
var MongoClient = mongo.MongoClient
  , assert = require('assert');

var GRAPH_STATIC_PATH   = path.resolve(__dirname, 'assets');
var HORIZON_STATIC_PATH = require('horizon-viz').staticFilePath() + '/assets/';

debug("Config set to %j", config);

// FIXME: Get real viz server public IP/DNS name from DB
var VIZ_SERVER_HOST = 'localhost';
// FIXME: Get real viz server port from DB
var VIZ_SERVER_PORT = config.LISTEN_PORT + 1;

var express = require('express'),
    app = express(),
    http = require('http').Server(app);

var db;

app.get('/vizaddr/graph', function(req, res) {
    if (config.PRODUCTION) {
        var name = req.param("dataName");
        name = "uber" // hardcode for now
        db.collection('data_info').findOne({"name": name}, function(err, doc) {
            if (err) {
                debug(err);
                res.send('Problem with query');
                res.end();
                return;
            }
            if (doc) {
                // Query only for gpus that have been updated within 30 secs
                var d = new Date();
                d.setSeconds(d.getSeconds() - 30);

                // Get all GPUs that have free memory that can fit the data
                db.collection('gpu_monitor')
                      .find({'gpu_memory_free': {'$gt': doc.size},
                             'updated': {'$gt': d}, },
                             {'sort': ['gpu_memory_free', 'desc']})
                      .toArray(function(err, ips) {

                    if (err) {
                        debug(err);
                        res.send('Problem with query');
                        res.end();
                        return;
                    }

                    // Are there no servers with enough space?
                    if (ips.length == 0) {
                        debug("All GPUs out of space!");
                        res.send('No servers can fit the data :/');
                        res.end();
                        return;
                    }

                    // Query only for workers that have been updated within 30 secs
                    var d = new Date();
                    d.setSeconds(d.getSeconds() - 30);

                    // Find all idle node processes
                    db.collection('node_monitor').find({'active': false, 
                                                        'updated': {'$gt': d}})
                                                     .toArray(function(err, results) {

                        if (err) {
                            debug(err);
                            res.send('Problem with query');
                            res.end();
                            return;
                        }

                        // Are all processes busy or dead?
                        if (results.length == 0) {
                            debug('There is space on a server, but all workers in the fleet are busy or dead (have not pinged home in over 30 seconds).');
                            res.send('There is space on a server, but all workers in the fleet are busy or dead (have not pinged home in over 30 seconds)');
                            res.end();
                            return;
                        }

                        // Try each IP in order of free space
                        for (var i in ips) {
                            var ip = ips[i]['ip'];

                            for (var j in results) {
                                if (results[j]['ip'] != ip) continue;

                                // We found a match
                                var port = results[j]['port'];

                                // Todo: ping process first for safety
                                debug("Assigning client '%s' to viz server on %s, port %d", req.ip, ip, port);
                                res.json({'hostname': ip, 'port': port});
                                res.end();
                                return;
                            }
                        }
                    });
                });
            } else {
                res.send('Couldn\'t find that dataset');
                res.end();
                return;
            }
        });
    } else {
        debug("Assigning client '%s' to viz server on %s, port %d", req.ip, VIZ_SERVER_HOST, VIZ_SERVER_PORT);
        res.json({'hostname': VIZ_SERVER_HOST, 'port': VIZ_SERVER_PORT});        
    }
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
app.use('/horizon', express.static(HORIZON_STATIC_PATH));
// Serve graph static assets (and /sc.html)
app.use(express.static(GRAPH_STATIC_PATH));


// Default '/' path redirects to graph demo
app.get('/', function(req, res) {
    debug('redirecting')
    res.redirect('/graph.html' + (req.query.debug !== undefined ? '?debug' : ''));
});


// Start listening for HTTP connections
MongoClient.connect(config.MONGO_SERVER, {auto_reconnect: true}, function(err, database) {
  if(err) debug(err);

  db = database.db('graphistry-prod');

  try {
      http.listen(3000, 'localhost', function() {
          console.log('\n[server.js] Server listening on %s:%d', 'localhost', 3000);
      });
  } catch(e) {
      console.error("[server.js] Fatal error: could not start server on address %s, port %s. Exiting...", 'localhost', 3000);
      process.exit(1);
  }
});