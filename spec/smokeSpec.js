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

// process.on('uncaughtException', function(err) {
//   console.log(err.stack);
//   throw err;
// });

function deepcopy (obj) {
    return JSON.parse(JSON.stringify(obj));
}

describe ("[SMOKE] Server-viz", function () {

    // Variables available to all tests
    var app;
    var buffernames;
    var client;
    var id;
    var options = {
        transports: ['websocket'],
        'force new connection': true,
        query: {datasetname: 'LayoutDebugLines'}
    };
    var socketURL = 'http://0.0.0.0:5432';
    var theRenderConfig;
    var vboBuffer = {};
    var lastVbos = {};

    // Setup
    it ("should setup app and connect", function (done) {
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
    it ("should get a render config", function (done) {
        client.on('render_config', function (render_config) {
            theRenderConfig = render_config;
            expect(render_config).toBeDefined();
            done();
        });
        client.emit('get_render_config');
    });

    it ("should start streaming and get an animation tick", function (done) {
        buffernames = renderer.getServerBufferNames(theRenderConfig);
        client.on('vbo_update', function (data) {
            for (var i = 0; i < buffernames.length; i++) {
                (function() {
                    var num = i;
                    supertest(app)
                        .get('/vbo')
                        .query({id: id})
                        .query({buffer: buffernames[num]})
                        .end(function (res) {
                            expect(res).toBeDefined();
                            vboBuffer[buffernames[num]] = res;
                            if (Object.keys(vboBuffer).length === buffernames.length) {
                                lastVbos = deepcopy(vboBuffer);
                                vboBuffer = {};
                                client.emit('received_buffers', 'faketime');
                                done();
                            }
                        });
                })();
            }
        });
        client.emit('begin_streaming');
        client.emit('animate');
    });


    // No support for afterAll, so using a test case to tear down
    it ("should tear down", function () {
        client.close();
        // TODO: Figure out how to actually tear down gracefully.
        // Since we can't easily end the loop right now, we just let jasmine
        // kill the process
    });


});
