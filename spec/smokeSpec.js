'use strict';

var Rx           = require('rx');
var _            = require('underscore');
var fs           = require('fs');
var path         = require('path');
var driver       = require('../js/node-driver.js');
var StreamGL     = require('StreamGL');
var compress     = require('node-pigz');
var renderer     = StreamGL.renderer;
var rConf        = require('../js/renderer.config.js');
var loader       = require('../js/data-loader.js');
var server       = require('../server-viz.js');
var config       = require('config')();
var io           = require('socket.io').listen(5432);
var ioClient     = require('socket.io-client');
var supertest    = require('supertest');
var express      = require('express');

// Because node swallows a lot of exceptions, uncomment this if tests are
// crashing without any details.
/*
process.on('uncaughtException', function(err) {
  console.log(err.stack);
  throw err;
});
*/

describe ("Smoke test using LayoutDebugLines", function () {

    // Variables available to all tests
    var app;
    var client;
    var id;
    var options = {
        transports: ['websocket'],
        'force new connection': true
    };
    var socketURL = 'http://0.0.0.0:5432';
    var theRenderConfig;

    // Setup
    it ("Should setup app and connect", function (done) {
        app = express();
        io.on('connection', function (socket) {
            socket.on('viz', function (msg, cb) { cb(); });
            server.init(app, socket);
        });
        client = ioClient.connect(socketURL, options);
        client.on('connect', function (data) {
            id = client.io.engine.id;
            done();
        });
    });

    // Tests
    it ("Should get a render config", function (done) {
        client.on('render_config', function (render_config) {
            theRenderConfig = render_config;
            expect(render_config).toBeDefined();
            done();
        });
        client.emit('get_render_config');
    });

    it ("Should start streaming and get 3 animation ticks", function (done) {
        var count = 1;
        client.on('vbo_update', function (data) {
            var buffernames = renderer.getServerBufferNames(theRenderConfig);
            var times = buffernames.length;
            for (var i = 0; i < times; i++) {
                (function() {
                    var num = i;
                    supertest(app)
                        .get('/vbo')
                        .query({id: id})
                        .query({buffer: buffernames[num]})
                        .end(function (res) {
                            expect(res).toBeDefined();
                            if (num + 1 === times) {
                                client.emit('animate');
                            }
                        });
                })();
            }
            client.emit('received_buffers', 'faketime');
            client.emit('animate');
            if (++count > 3) {
                done();
            }
        });
        client.emit('begin_streaming');
        client.emit('animate');
    });


    // No support for afterAll, so using a test case to tear down
    it ("Should tear down", function () {
        client.close();
        // TODO: Figure out how to actually tear down gracefully.
        // Since we can't easily end the loop right now, we just let jasmine
        // kill the process
    });


});
