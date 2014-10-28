#!/usr/bin/env node
'use strict';

//Set jshint to ignore `predef:'io'` in .jshintrc so we can manually define io here
/* global -io */

var config      = require('./config')();

var Rx          = require('rx'),
    _           = require('underscore'),
    debug       = require('debug')('StreamGL:server'),
    fs          = require('fs'),
    path        = require('path');

var driver      = require('./js/node-driver.js'),
    compress    = require('node-pigz'),
    renderer    = require(path.resolve(config.STREAMGL_PATH, 'renderer.js'));


debug("Config set to %j", config);


var express = require('express'),
    app = express(),
    http = require('http').Server(app),
    io = require('socket.io')(http, {transports: ['websocket']});

//FIXME CHEAP HACK TO WORK AROUND CONFIG FILE INCLUDE PATH
var cwd = process.cwd();
process.chdir(path.resolve(config.GPU_STREAMING_PATH, 'StreamGL'));
var renderConfig = require(path.resolve(config.STREAMGL_PATH, 'renderer.config.graph.js'));
process.chdir(cwd);


/**** GLOBALS ****************************************************/

// ----- BUFFERS (multiplexed over clients) ----------
//Serve most recent compressed binary buffers
//TODO reuse across users
//{socketID -> {buffer...}
var lastCompressedVbos;
var finishBufferTransfers;


// ----- ANIMATION ------------------------------------
//current animation
var animStep;

//multicast of current animation's ticks
var ticksMulti;

//most recent tick
var graph;


// ----- INITIALIZATION ------------------------------------

//Do more innocuous initialization inline (famous last words..)

function resetState () {
    debug('RESETTING APP STATE');

    //FIXME explicitly destroy last graph if it exists?

    lastCompressedVbos = {};
    finishBufferTransfers = {};


    animStep = driver.create();
    ticksMulti = animStep.ticks.publish();
    ticksMulti.connect();

    //make available to all clients
    graph = new Rx.ReplaySubject(1);
    ticksMulti.take(1).subscribe(graph, debug.bind('ERROR ticksMulti'));

    debug('RESET APP STATE.');
}


resetState();


/**** END GLOBALS ****************************************************/



/** Given an Object with buffers as values, returns the sum size in megabytes of all buffers */
function vboSizeMB(vbos) {
    var vboSizeBytes =
        _.reduce(
            _.pluck(_.values(vbos.buffers), 'byteLength'),
            function(acc, v) { return acc + v; }, 0);
    return (vboSizeBytes / (1024 * 1024)).toFixed(1);
}


// Express middleware function for sending "don't cache" headers to the browser
function nocache(req, res, next) {
    res.header('Cache-Control', 'private, no-cache, no-store, must-revalidate');
    res.header('Expires', '-1');
    res.header('Pragma', 'no-cache');
    next();
}
app.use(nocache);



app.get('/vbo', function(req, res) {
    debug('VBOs: HTTP GET %s', req.originalUrl, req.query);

    try {
        // TODO: check that query parameters are present, and that given id, buffer exist
        var bufferName = req.query.buffer;
        var id = req.query.id;

        res.set('Content-Encoding', 'gzip');

        res.send(lastCompressedVbos[id][bufferName]);
    } catch (e) {
        console.error('bad request', e, e.stack);
    }

    finishBufferTransfers[id](bufferName);
});

var colorTexture = new Rx.ReplaySubject(1);
var img =
    Rx.Observable.fromNodeCallback(fs.readFile)('test-colormap2.rgba')
    .flatMap(function (buffer) {
        debug('Loaded raw colorTexture', buffer.length);
        return Rx.Observable.fromNodeCallback(compress.deflate)(
                buffer,//binary,
                {output: new Buffer(
                    Math.max(1024, Math.round(buffer.length * 1.5)))})
            .map(function (compressed) {
                return {
                    raw: buffer,
                    compressed: compressed
                };
            });
    })
    .do(function () { debug('Compressed color texture'); })
    .map(function (pair) {
        debug('colorMap bytes', pair.raw.length);
        return {
            buffer: pair.compressed[0],
            bytes: pair.raw.length,
            width: 512,
            height: 512
        };
    });

img.take(1).subscribe(colorTexture, debug.bind('ERROR IMG'));
colorTexture.subscribe(
    function() { debug('HAS COLOR TEXTURE'); },
    debug.bind('ERROR colorTexture'));



app.get('/vbo', function(req, res) {
    debug('VBOs: HTTP GET %s', req.originalUrl);

    try {
        // TODO: check that query parameters are present, and that given id, buffer exist
        var bufferName = req.query.buffer;
        var id = req.query.id;

        res.set('Content-Encoding', 'gzip');
        res.send(lastCompressedVbos[id][bufferName]);

    } catch (e) {
        console.error('bad request', e, e.stack);
    }

    finishBufferTransfers[id](bufferName);
});

app.get('/texture', function (req, res) {
    debug('got texture req', req.originalUrl, req.query);
    try {

        var textureName = req.query.texture;
        var id = req.query.id;

        colorTexture.pluck('buffer').subscribe(
            function (data) {
                res.set('Content-Encoding', 'gzip');
                res.send(data);
            },
            debug.bind('ERROR colorTexture pluck'));

    } catch (e) {
        console.error('bad request', e, e.stack);
    }
});



io.on('connection', function(socket) {
    debug('Client connected', socket.id);

    // ========== BASIC COMMANDS

    lastCompressedVbos[socket.id] = {};
    socket.on('disconnect', function () {
        debug('disconnecting', socket.id);
        delete lastCompressedVbos[socket.id];
    });



    //Used for tracking what needs to be sent
    //Starts as all active, and as client caches, whittles down
    var activeBuffers = renderer.getServerBufferNames(renderConfig),
        activeTextures = renderer.getServerTextureNames(renderConfig),
        activePrograms = renderConfig.scene.render;

    var requestedBuffers = activeBuffers,
        requestedTextures = activeTextures;

    //Knowing this helps overlap communication and computations
    socket.on('planned_binary_requests', function (request) {
        debug('CLIENT SETTING PLANNED REQUESTS', request.buffers, request.textures);
        requestedBuffers = request.buffers;
        requestedTextures = request.textures;
    });


    debug('active buffers/textures/programs', activeBuffers, activeTextures, activePrograms);

    socket.on('graph_settings', function (payload) {
        debug('new settings', payload, socket.id);
        animStep.proxy(payload);
    });

    socket.on('reset_graph', function (_, cb) {
        debug('reset_graph command');
        resetState();
        cb();
    });



    // ============= EVENT LOOP

    //starts true, set to false whenever transfer starts, true again when ack'd
    var clientReady = new Rx.ReplaySubject(1);
    clientReady.onNext(true);
    socket.on('received_buffers', function (time) {
        debug('Client end-to-end time', time);
        clientReady.onNext(true);
    });

    clientReady.subscribe(debug.bind('CLIENT STATUS'), debug.bind('ERROR clientReady'));

    debug('SETTING UP CLIENT EVENT LOOP');
    var step = 0;
    graph.expand(function (graph) {
        step++;

        debug('1. Prefetch VBOs', socket.id, activeBuffers);

        return driver.fetchData(graph, compress, activeBuffers, activePrograms)
            .do(function (vbos) {
                debug('prefetched VBOs for xhr2: ' + vboSizeMB(vbos.compressed) + 'MB');
                //tell XHR2 sender about it
                lastCompressedVbos[socket.id] = vbos.compressed;
            })
            .flatMap(function (vbos) {
                debug('2. Waiting for client to finish previous', socket.id);
                return clientReady
                    .filter(_.identity)
                    .take(1)
                    .do(function () {
                        debug('2b. Client ready, proceed and mark as processing.', socket.id);
                        clientReady.onNext(false);
                    })
                    .map(_.constant(vbos));
            })
            .flatMap(function (vbos) {
                debug('3. tell client about availablity', socket.id);

                //for each buffer transfer
                var sendingAllBuffers = new Rx.Subject();
                var clientAckStartTime;
                var clientElapsed;
                var transferredBuffers = [];
                finishBufferTransfers[socket.id] = function (bufferName) {
                    debug('3a ?. sending a buffer', bufferName, socket.id);
                    transferredBuffers.push(bufferName);
                    if (transferredBuffers.length === requestedBuffers.length) {
                        debug('3b. started sending all', socket.id);
                        debug('Socket', '...client ping ' + clientElapsed + 'ms');
                        debug('Socket', '...client asked for all buffers',
                            Date.now() - clientAckStartTime, 'ms');
                        sendingAllBuffers.onNext();
                    }
                };

                var emitFnWrapper = Rx.Observable.fromCallback(socket.emit, socket);

                //notify of buffer/texture metadata
                //FIXME make more generic and account in buffer notification status
                colorTexture.flatMap(function (colorTexture) {
                        debug('unwrapped texture meta');

                        var textures = {
                            colorMap: _.pick(colorTexture, ['width', 'height', 'bytes'])
                        };

                        //FIXME: should show all active VBOs, not those based on prev req
                        var metadata =
                            _.extend(
                                _.pick(vbos, ['bufferByteLengths', 'elements']),
                                {textures: textures,
                                 versions: {
                                    buffers: vbos.versions,
                                    textures: {
                                        colorMap: 1
                                    }
                                }});

                        debug('notifying client of buffer metadata', metadata);
                        return emitFnWrapper('vbo_update', metadata);

                    }).subscribe(
                        function (clientElapsedMsg) {
                            debug('3d ?. client all received', socket.id);
                            clientElapsed = clientElapsedMsg;
                            clientAckStartTime = Date.now();
                        },
                        debug.bind('ERROR SENDING METADATA'));

                return sendingAllBuffers
                    .take(1)
                    .do(debug.bind('3c. All in transit', socket.id));
            })
            .flatMap(function () {
                debug('4. Wait for next anim step', socket.id);
                return ticksMulti
                    .take(1)
                    .do(function () { debug('4b. next ready!', socket.id); });
            })
            .map(_.constant(graph));
    })
    .subscribe(function () { debug('LOOP ITERATED', socket.id); }, debug.bind('ERROR LOOP'));

});


app.use(express.static(config.GPU_STREAMING_PATH));

http.listen(config.LISTEN_PORT, config.LISTEN_ADDRESS, function() {
    console.log('\nServer listening on %s:%d', config.LISTEN_ADDRESS, config.LISTEN_PORT);
});
