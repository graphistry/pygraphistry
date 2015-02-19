#!/usr/bin/env node
'use strict';

//Set jshint to ignore `predef:'io'` in .jshintrc so we can manually define io here
/* global -io */

var Rx          = require('rx');
var _           = require('underscore');
var debug       = require('debug')('graphistry:graph-viz:driver:viz-server');
var profiling   = require('debug')('profiling');
var fs          = require('fs');
var path        = require('path');
var rConf       = require('./js/renderer.config.js');
var loader      = require('./js/data-loader.js');
var driver      = require('./js/node-driver.js');
var compress    = require('node-pigz');
var StreamGL    = require('StreamGL');
var config      = require('config')();

var renderer = StreamGL.renderer;


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

function resetState(dataset) {
    debug('RESETTING APP STATE');

    //FIXME explicitly destroy last graph if it exists?

    lastCompressedVbos = {};
    finishBufferTransfers = {};


    animStep = driver.create(dataset);
    ticksMulti = animStep.ticks.publish();
    ticksMulti.connect();

    //make available to all clients
    graph = new Rx.ReplaySubject(1);
    ticksMulti.take(1).subscribe(graph, debug.bind('ERROR ticksMulti'));

    debug('RESET APP STATE.');
}


function getState() {
    return animStep.graph.then(function (graph) {
        return graph;
    })
}

/**** END GLOBALS ****************************************************/




function makeErrorHandler(name) {
    return function (err) {
        console.error(name, err, (err||{}).stack);
    };
}


/** Given an Object with buffers as values, returns the sum size in megabytes of all buffers */
function vboSizeMB(vbos) {
    var vboSizeBytes =
        _.reduce(
            _.pluck(_.values(vbos.buffers), 'byteLength'),
            function(acc, v) { return acc + v; }, 0);
    return (vboSizeBytes / (1024 * 1024)).toFixed(1);
}


function init(app, socket) {
    debug('Client connected', socket.id);

    var colorTexture = new Rx.ReplaySubject(1);
    var imgPath = path.resolve(__dirname, 'test-colormap2.rgba');
    var img =
        Rx.Observable.fromNodeCallback(fs.readFile)(imgPath)
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

    img.take(1)
        .do(colorTexture)
        .subscribe(_.identity, makeErrorHandler('ERROR IMG'));
    colorTexture
        .do(function() { debug('HAS COLOR TEXTURE'); })
        .subscribe(_.identity, makeErrorHandler('ERROR colorTexture'));



    app.get('/vbo', function(req, res) {
        debug('VBOs: HTTP GET %s', req.originalUrl);
        profiling('VBO request');

        try {
            // TODO: check that query parameters are present, and that given id, buffer exist
            var bufferName = req.query.buffer;
            var id = req.query.id;

            res.set('Content-Encoding', 'gzip');
            var vbos = lastCompressedVbos[id];
            if (vbos) {
                res.send(lastCompressedVbos[id][bufferName]);
            }
            res.send();

        } catch (e) {
            console.error('[viz-server.js] bad request', e, e.stack);
        }

        finishBufferTransfers[id](bufferName);
    });

    app.get('/texture', function (req, res) {
        debug('got texture req', req.originalUrl, req.query);
        try {
            colorTexture.pluck('buffer').do(
                function (data) {
                    res.set('Content-Encoding', 'gzip');
                    res.send(data);
                })
                .subscribe(_.identity,
                    makeErrorHandler('ERROR colorTexture pluck'));

        } catch (e) {
            console.error('[viz-server.js] bad request', e, e.stack);
        }
    });

    // Get the datasetname from the socket query param, sent by Central
    var datasetName = socket.handshake.query.dataset || config.DATASETNAME || 'Uber';
    var theDataset = loader.downloadDataset(datasetName);

    var theRenderConfig = theDataset.then(function (dataset) {
        var config = dataset.Metadata.config;
        var query = socket.handshake.query;

        // URL parameters override config provided by the dataset
        function hasParam(param) { return param !== undefined && param !== 'undefined' }
        config.scene    = hasParam(query.scene)    ? query.scene    : config.scene;
        config.controls = hasParam(query.controls) ? query.controls : config.controls;
        config.mapper   = hasParam(query.mapper)   ? query.mapper   : config.mapper;
        config.device   = hasParam(query.device)   ? query.device   : config.device;

        console.info('scene:%s  controls:%s  mapper:%s  device:%s',
                     config.scene, config.controls, config.mapper, config.device);

        if (!(config.scene in rConf.scenes)) {
            console.warn('WARNING Unknown scene "%s", using default', config.scene)
            config.scene = 'default';
        }

        resetState(dataset);
        return rConf.scenes[config.scene];
    }).fail(function (err) {
        console.error('ERROR in initialization: ', (err||{}).stack);
    });

    socket.on('get_render_config', function() {
        debug('Sending render-config to client');
        theRenderConfig.then(function (renderConfig) {
            socket.emit('render_config', renderConfig);
        }).fail(function (err) {
            console.error('ERROR sending rendererConfig ', (err||{}).stack);
        });
    });

    socket.on('begin_streaming', function() {
        theRenderConfig.then(function (renderConfig) {
            stream(socket, renderConfig, colorTexture);
        }).fail(function (err) {
            console.error('ERROR streaming ', (err||{}).stack);
        });
    });

    socket.on('reset_graph', function (_, cb) {
        debug('reset_graph command');
        theDataset.then(function (dataset) {
            resetState(dataset);
            cb();
        }).fail(function (err) {
            console.error('ERROR resetting graph ', (err||{}).stack);
        });
    });

    return module.exports;
}

function stream(socket, renderConfig, colorTexture) {

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
        activePrograms = renderConfig.render;

    var requestedBuffers = activeBuffers,
        requestedTextures = activeTextures;

    //Knowing this helps overlap communication and computations
    socket.on('planned_binary_requests', function (request) {
        debug('CLIENT SETTING PLANNED REQUESTS', request.buffers, request.textures);
        requestedBuffers = request.buffers;
        requestedTextures = request.textures;
    });


    debug('active buffers/textures/programs', activeBuffers, activeTextures, activePrograms);


    socket.on('interaction', function (payload) {
        profiling('Got Interaction');
        debug('Got interaction:', payload);
        // TODO: Find a way to avoid flooding main thread waiting for GPU ticks.
        var defaults = {play: false, layout: false};
        animStep.interact(_.extend(defaults, payload || {}));
    });

    socket.on('get_labels', function (labels, cb) {
        graph.take(1)
            .do(function (graph) {
                var offset = graph.simulator.timeSubset.pointsRange.startIdx;
                var hits = labels.map(function (idx) { return graph.simulator.labels[offset + idx]; });
                cb(null, hits);
            })
            .subscribe(_.identity, makeErrorHandler('get_labels'));
    });




    // ============= EVENT LOOP

    //starts true, set to false whenever transfer starts, true again when ack'd
    var clientReady = new Rx.ReplaySubject(1);
    clientReady.onNext(true);
    socket.on('received_buffers', function (time) {
        profiling('Received buffers');
        debug('Client end-to-end time', time);
        clientReady.onNext(true);
    });

    clientReady.subscribe(debug.bind('CLIENT STATUS'), debug.bind('ERROR clientReady'));

    debug('SETTING UP CLIENT EVENT LOOP ===================================================================');
    var step = 0;
    var lastVersions = null;
    graph.expand(function (graph) {
        step++;

        var ticker = {step: step};

        debug('0. Prefetch VBOs', socket.id, activeBuffers, ticker);

        return driver.fetchData(graph, renderConfig, compress,
                                activeBuffers, lastVersions, activePrograms)
            .do(function (vbos) {
                debug('1. prefetched VBOs for xhr2: ' + vboSizeMB(vbos.compressed) + 'MB', ticker);
                //tell XHR2 sender about it
                lastCompressedVbos[socket.id] = vbos.compressed;
            })
            .flatMap(function (vbos) {
                debug('2. Waiting for client to finish previous', socket.id, ticker);
                return clientReady
                    .filter(_.identity)
                    .take(1)
                    .do(function () {
                        debug('2b. Client ready, proceed and mark as processing.', socket.id, ticker);
                        clientReady.onNext(false);
                    })
                    .map(_.constant(vbos));
            })
            .flatMap(function (vbos) {
                debug('3. tell client about availablity', socket.id, ticker);

                //for each buffer transfer
                var sendingAllBuffers = new Rx.Subject();
                var clientAckStartTime;
                var clientElapsed;
                var transferredBuffers = [];
                finishBufferTransfers[socket.id] = function (bufferName) {
                    debug('5a ?. sending a buffer', bufferName, socket.id, ticker);
                    transferredBuffers.push(bufferName);
                    if (transferredBuffers.length === requestedBuffers.length) {
                        debug('5b. started sending all', socket.id, ticker);
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
                        debug('4a. unwrapped texture meta', ticker);

                        var textures = {
                            colorMap: _.pick(colorTexture, ['width', 'height', 'bytes'])
                        };

                        //FIXME: should show all active VBOs, not those based on prev req
                        var metadata =
                            _.extend(
                                _.pick(vbos, ['bufferByteLengths', 'elements']),
                                {
                                    textures: textures,
                                    versions: {
                                        buffers: vbos.versions,
                                        textures: {colorMap: 1}},
                                    step: step
                                });
                        lastVersions = vbos.versions;

                        debug('4b. notifying client of buffer metadata', metadata, ticker);
                        profiling('===Sending VBO Update===');
                        return emitFnWrapper('vbo_update', metadata);

                    }).do(
                        function (clientElapsedMsg) {
                            debug('6 ?. client all received', socket.id, ticker);
                            clientElapsed = clientElapsedMsg;
                            clientAckStartTime = Date.now();
                        })
                    .subscribe(_.identity, makeErrorHandler('ERROR SENDING METADATA'));

                return sendingAllBuffers
                    .take(1)
                    .do(debug.bind('7. All in transit', socket.id, ticker));
            })
            .flatMap(function () {
                debug('8. Wait for next anim step', socket.id, ticker);
                return ticksMulti
                    .take(1)
                    .do(function () { debug('9. next ready!', socket.id, ticker); });
            })
            .map(_.constant(graph));
    })
    .subscribe(function () { debug('10. LOOP ITERATED', socket.id); }, makeErrorHandler('ERROR LOOP'));
}


if (require.main === module) {

    var express = require('express'),
        app     = express(),
        http    = require('http').Server(app),
        io      = require('socket.io')(http, {transports: ['websocket']});

    debug('Config set to %j', config);

    var nocache = function (req, res, next) {
        res.header('Cache-Control', 'private, no-cache, no-store, must-revalidate');
        res.header('Expires', '-1');
        res.header('Pragma', 'no-cache');
        next();
    };
    app.use(nocache);

    var allowCrossOrigin = function  (req, res, next) {
        res.header('Access-Control-Allow-Origin', '*');
        res.header('Access-Control-Allow-Headers', 'X-Requested-With,Content-Type,Authorization');
        res.header('Access-Control-Allow-Methods', 'GET,PUT,PATCH,POST,DELETE');
        next();
    };
    app.use(allowCrossOrigin);

    //Static assets
    app.get('*/StreamGL.js', function(req, res) {
        res.sendFile(require.resolve('StreamGL/dist/StreamGL.js'));
    });
    app.get('*/StreamGL.map', function(req, res) {
        res.sendFile(require.resolve('StreamGL/dist/StreamGL.map'));
    });
    app.use('/graph', function (req, res, next) {
        return express.static(path.resolve(__dirname, 'assets'))(req, res, next);
    });

    //Dyn routing
    app.get('/vizaddr/graph', function(req, res) {
        res.json({'hostname': config.HTTP_LISTEN_ADDRESS, 'port': config.HTTP_LISTEN_PORT});
    });


    io.on('connection', function (socket) {
        socket.on('viz', function (msg, cb) { cb({succeed: true}); });
        init(app, socket);
    });

    var listen = Rx.Observable.fromNodeCallback(
            http.listen.bind(http, config.HTTP_LISTEN_PORT, config.HTTP_LISTEN_ADDRESS))();

    listen.subscribe(
            function () { console.log('\nViz worker listening'); },
            function (err) { console.error('\nError starting viz worker', err); });

}


module.exports = {
    init: init,
    getState: getState
}
