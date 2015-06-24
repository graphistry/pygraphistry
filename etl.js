'use strict';

var urllib   = require('url');
var zlib     = require('zlib');
var debug    = require('debug')('graphistry:etlworker:etl');
var _        = require('underscore');
var Q        = require('q');
var bodyParser  = require('body-parser');

var config   = require('config')();

var vgraph   = require('./vgraph.js');
var Cache    = require('common/cache.js');
var s3       = require('common/s3.js');

var tmpCache = new Cache(config.LOCAL_CACHE_DIR, config.LOCAL_CACHE);

// Convert JSON edgelist to VGraph then upload VGraph to S3 and local /tmp
// JSON
function etl(msg) {
    var name = decodeURIComponent(msg.name);
    debug('ETL for', msg.name);
    //debug('Data', msg.labels);

    var vg = vgraph.fromEdgeList(
        msg.graph,
        msg.labels,
        msg.bindings.sourceField,
        msg.bindings.destinationField,
        msg.bindings.idField,
        name
    );

    if (vg === undefined) {
        throw new Error('Invalid edgelist');
    } else {
        return publish(vg, name);
    }
}


// VGraph * String -> Promise[String]
function publish(vg, name) {
    var metadata = {name: name || vg.name};
    var binData = vg.encode().toBuffer();

    function cacheLocally() {
        // Wait a couple of seconds to make sure our cache has a
        // more recent timestamp than S3
        var res = Q.defer();
        setTimeout(function () {
            debug('Caching dataset locally');
            res.resolve(tmpCache.put(urllib.parse(name), binData));
        }, 2000);
        return res.promise;
    }

    if (config.ENVIRONMENT === 'local') {
        debug('Attempting to upload dataset');
        return s3Upload(binData, metadata)
            .fail(function (err) {
                console.error('S3 Upload failed', err.message);
            }).then(cacheLocally, cacheLocally) // Cache locally regardless of result
            .then(_.constant(name)); // We succeed iff cacheLocally succeeds
    } else {
        // On prod/staging ETL fails if upload fails
        debug('Uploading dataset');
        return s3Upload(binData, metadata)
            .then(_.constant(name))
            .fail(function (err) {
                console.error('S3 Upload failed', err.message);
            });
    }
}


// Buffer * {name: String, ...} -> Promise
function s3Upload(binaryBuffer, metadata) {
    return s3.upload(config.S3, config.BUCKET, metadata, binaryBuffer)
}


function req2data(req) {
    var encoding = req.headers['content-encoding'] || 'identity'
    var chunks = [];
    var result = Q.defer();

    req.on('data', function (chunk) {
        chunks.push(chunk);
    });

    req.on('end', function () {
        var data = Buffer.concat(chunks)

        debug('Request bytes:%d, encoding:%s', data.length, encoding);

        if (encoding == 'identity') {
            result.resolve(data.toString());
        } else if (encoding === 'gzip') {
            result.resolve(Q.denodeify(zlib.gunzip)(data))
        } else if (encoding === 'deflate') {
            result.resolve(Q.denodeify(zlib.inflate)(data))
        } else {
            result.reject('Unknown encoding: ' + encoding)
        }
    });

    return result.promise;
}


function makeFailHandler(res) {
    return function (err) {
        console.error('ETL post fail', (err||{}).stack);
        res.send({
            success: false,
            msg: err.message
        });
        console.error('Failed worker, exiting');
        process.exit(1);
    }
}


// Handler for ETL requests on central/etl
function jsonEtl(k, req, res) {
    req2data(req).then(function (data) {
        try {
            etl(JSON.parse(data))
                .then(function (name) {
                    debug('ETL done, notifying client to proceed');
                    res.send({ success: true, dataset: name });
                    k();
                }, makeFailHandler(res));
        } catch (err) {
            makeFailHandler(res)(err);
        }
    }).fail(makeFailHandler(res));
}


function vgraphEtl(k, req, res) {
    return req2data(req).then(function (data) {
            try {
                var buffer = new Buffer(data);
                var vg = vgraph.decodeVGraph(buffer);
                publish(vg);
                k();
            } catch (err) {
                makeFailHandler(res)(err)
            }
        }).fail(makeFailHandler(res));
}


function route (app, socket) {
    var done = function () {
        debug('worker finished, exiting');
        if (config.ENVIRONMENT === 'production' || config.ENVIRONMENT === 'staging') {
            process.exit(0);
        } else {
            console.warn('not actually exiting, only disconnect socket');
            socket.disconnect();
        }
    };

    app.post('/etl', bodyParser.json({type: '*', limit: '64mb'}), jsonEtl.bind('', done));
    app.post('/etlvgraph', bodyParser.raw({type: '*', limit: '64mb'}), vgraphEtl.bind('', done))
}


module.exports = {
    route: route
}
