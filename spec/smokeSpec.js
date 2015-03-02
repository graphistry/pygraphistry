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
var ioClient     = require('socket.io-client');
var supertest    = require('supertest');
var express      = require('express');
var app          = express();
var http         = require('http').Server(app);
var XMLHttpRequest = require('xhr2');
var io           = require('socket.io')(http, {transports: ['websocket']});
var zlib         = require('zlib');

// Because node swallows a lot of exceptions, uncomment this if tests are
// crashing without any details.

process.on('uncaughtException', function(err) {
  console.log(err.stack);
  throw err;
});

function distance(x, y) {
    return Math.sqrt(Math.pow(x[0] - y[0], 2) + Math.pow(x[1] - y[1], 2))
}

describe ("[SMOKE] Server-viz", function () {
    var animatePayload = {play: true, layout: true};
    var buffernames;
    var clients = {};
    var uberClient;
    var ids = {};
    var layoutOptions = {
        transports: ['websocket'],
        'force new connection': true,
        query: {dataset: 'LayoutDebugLines', scene: 'netflow', type: 'vgraph'}
    };
    var uberOptions = {
        transports: ['websocket'],
        'force new connection': true,
        query: {dataset: 'Uber', controls: 'uber', scene: 'uber', type: 'OBSOLETE_geo'}
    };
    var socketURL = 'http://localhost:3000';
    var appURL = 'http://localhost:3000';
    var theRenderConfig;
    var vboBuffer = {};
    var lastVbos = {};

    var processVbos = function (data, handshake, names, cb, client, id) {
        var lengths = data.bufferByteLengths;
        _.each(_.range(names.length), function (i) {
            var name = names[i];
            var oReq = new XMLHttpRequest();
            var getUrl = appURL + '/' + 'vbo?buffer' + '=' + name + '&id=' + id;
            oReq.open('GET', getUrl, true);
            oReq.responseType = 'arraybuffer';
            oReq.onload = function () {
                // We do a conversion here because zlib.gunzip expects a nodejs Buffer.
                var bufferResponse = new Buffer( new Uint8Array( oReq.response));
                zlib.gunzip(bufferResponse, function(err, result) {
                    var trimmedArray = new Uint8Array(result, 0, lengths[name]);
                    vboBuffer[name] = trimmedArray;
                    if (_.keys(vboBuffer).length === names.length) {
                        lastVbos = _.extend({}, vboBuffer);
                        vboBuffer = {};
                        client.emit('received_buffers', 'faketime');
                        cb();
                    }
                });
            };
            oReq.send();
        });
    }

    // Setup
    // TODO: Consider having this callable in the server-viz itself
    it ("should setup http and socket listener", function (done) {
        var listen = Rx.Observable.fromNodeCallback(
                http.listen.bind(http, config.HTTP_LISTEN_PORT, config.HTTP_LISTEN_ADDRESS))();
        listen.subscribe(
                function () { console.log('\nViz worker listening'); done();},
                function (err) { console.error('\nError starting viz worker', err); });
        io.on('connection', function (socket) {
            console.log("Connected");
            socket.on('viz', function (msg, cb) { cb(); });
            server.init(app, socket);
        });
    });

    it ("should setup app and connect to LayoutDebugLines", function (done) {
        clients.layout = ioClient.connect(socketURL, layoutOptions);
        clients.layout.on('connect', function (data) {
            ids.layout = clients.layout.io.engine.id;
            done();
        });
    });

    // Tests
    it ("should get correct render config for LayoutDebugLines", function (done) {
        clients.layout.on('render_config', function (render_config) {
            theRenderConfig = render_config;
            expect(render_config).toBeDefined();
            var render_list = ['pointpicking', 'pointsampling',
                'edgeculled', 'pointoutline', 'pointculled'];
            expect(render_config.render).toEqual(render_list);
            var program_list = ['points', 'edgeculled', 'pointculled'];
            expect(_.keys(render_config.programs)).toEqual(program_list);
            done();
        });
        clients.layout.emit('get_render_config');
    });

    it ("should start streaming LayoutDebugLines and get an animation tick", function (done) {
        clients.layout.on('vbo_update', function (data, handshake) {
            buffernames = renderer.getServerBufferNames(theRenderConfig);
            processVbos(data, handshake, buffernames, done, clients.layout, ids.layout);
        });
        clients.layout.emit('begin_streaming');
        setTimeout(function () {
            clients.layout.emit('interaction', animatePayload);
        }, 100);
    });

    it ("should have returned initial vbos of correct size for 8 points", function () {
        // Float, count=2, stride=8, DEVICE
        var curPoints = new Float32Array(lastVbos.curPoints.buffer);
        expect(curPoints.length).toBe(16);

        // Uint8, count=1, stride=0, HOST
        var pointSizes = lastVbos.pointSizes;
        expect(pointSizes.length).toBe(8);

        // Float, count=2, stride=8, DEVICE
        var springsPos = new Float32Array(lastVbos.springsPos.buffer);
        expect(springsPos.length).toBe(16);

        // Uint8, count=4, stride=0, HOST
        var edgeColors = lastVbos.edgeColors;
        expect(edgeColors.length).toBe(32);

        // Uint8, count=4, stride=0, HOST
        var pointColors = lastVbos.pointColors;
        expect(pointColors.length).toBe(32);
    });

    xit ("should converge positions after 50 iterations", function (done) {
        jasmine.getEnv().defaultTimeoutInterval = 10000;
        var iterations = 0;
        var cb = function() {
            if (++iterations >= 50) {
                var curPoints = new Float32Array(lastVbos.curPoints.buffer);
                _.each(_.range(4), function (i) {
                    var points = curPoints.slice(4*i, 4*(i+1));
                    var p1 = points.slice(0,2);
                    var p2 = points.slice(2,4);
                    var dist = distance(p1, p2);
                    expect(dist).toBeLessThan(0.02);
                });
                clients.layout.removeAllListeners('vbo_update');
                done();
            } else {
                clients.layout.emit('interaction', animatePayload);
            }
        }
        clients.layout.removeAllListeners('vbo_update');
        clients.layout.on('vbo_update', function (data, handshake) {
            processVbos(data, handshake, buffernames, cb, clients.layout, ids.layout);
        });
        clients.layout.emit('interaction', animatePayload);

    });

    it ("should connect to Uber dataset", function (done) {
        clients.uber = ioClient.connect(socketURL, uberOptions);
        clients.uber.on('connect', function (data) {
            ids.uber = clients.uber.io.engine.id;
            done();
        });
    });

    it ("should get correct render config for Uber", function (done) {
        clients.uber.on('render_config', function (render_config) {
            theRenderConfig = render_config;
            expect(render_config).toBeDefined();
            var render_list = ['pointpicking', 'pointsampling',
                'midedgetextured', 'pointculled'];
            expect(render_config.render).toEqual(render_list);
            var program_list = ['points', 'midedgetextured', 'pointculled'];
            expect(_.keys(render_config.programs)).toEqual(program_list);
            done();
        });
        clients.uber.emit('get_render_config');
    });

    it ("should start streaming Uber and get an animation tick", function (done) {
        clients.uber.on('vbo_update', function (data, handshake) {
            buffernames = renderer.getServerBufferNames(theRenderConfig);
            processVbos(data, handshake, buffernames, done, clients.uber, ids.uber);
        });
        clients.uber.emit('begin_streaming');
        setTimeout(function () {
            clients.uber.emit('interaction', animatePayload);
        }, 100);
    });

    // No support for afterAll, so using a test case to tear down
    it ("should tear down", function () {
        _.each(clients, function (client) {
            client.close();
        });
        // TODO: Figure out how to actually tear down gracefully.
        // Since we can't easily end the loop right now, we just let jasmine
        // kill the process
    });


});
