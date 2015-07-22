'use strict';

var urllib   = require('url');
var zlib     = require('zlib');
var _        = require('underscore');
var Q        = require('q');
var bodyParser  = require('body-parser');
var slack    = require('slack-write');

var config   = require('config')();

var vgraph   = require('./vgraph.js');
var Cache    = require('common/cache.js');
var s3       = require('common/s3.js');
var apiKey   = require('common/api.js');

var Log         = require('common/logger.js');
var logger      = Log.createLogger('etlworker:etl');

var tmpCache = new Cache(config.LOCAL_CACHE_DIR, config.LOCAL_CACHE);

// String * String -> ()
function slackNotify(name, params, nnodes, nedges) {
    function makeUrl(server) {
        return '<http://proxy-' + server + '.graphistry.com' +
               '/graph/graph.html?info=true&dataset=' + name +
               '|' + server + '>';
    }

    var slackConf = {
        token: config.SLACK_BOT_ETL_TOKEN,
        channel: '#datasets',
        username: 'etl'
    };

    var user = params.usertag.split('-')[0];
    var part1 = 'New dataset *' + name + '* by user `' + user + '`\n' +
                'Nodes: ' + nnodes + ',    Edges: ' + nedges + '\n';
    var part2 = '_Agent_: ' + params.agent + ',    ' +
                '_AgentVersion_: ' + params.agentVersion + ',    ' +
                '_API_: ' + params.apiVersion + '\n';
    var part3 = 'View on ' + makeUrl('labs') + ' or ' + makeUrl('staging') + '\n';

    var part4 = 'Key: ';
    if (params.key) {
        try {
            part4 += apiKey.decrypt(params.key);
        } catch (err) {
            logger.error('Could not decrypt key', err);
            part4 += ' COULD NOT DECRYPT';
        }
    } else {
        part4 = 'Key: n/a';
    }

    if (slackConf.token == undefined) {
        return Q();
    } else {
        return Q.denodeify(slack.write)(part1 + part2 + part3 + part4, slackConf)
            .fail(function (err) {
                logger.error('Error posting on slack', err);
            });
    }
}


// Convert JSON edgelist to VGraph then upload VGraph to S3 and local /tmp
// JSON
function etl(msg, params) {
    var name = decodeURIComponent(msg.name);
    logger.debug('ETL for', msg.name);

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
    }

    logger.info('VGraph created with', vg.nvertices, 'nodes and', vg.nedges, 'edges');
    return Q.all([
        publish(vg, name),
        slackNotify(name, params, vg.nvertices, vg.nedges)
    ]).spread(_.identity);
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
            logger.debug('Caching dataset locally');
            res.resolve(tmpCache.put(urllib.parse(name), binData));
        }, 2000);
        return res.promise;
    }

    if (config.ENVIRONMENT === 'local') {
        logger.debug('Attempting to upload dataset');
        return s3Upload(binData, metadata)
            .fail(function (err) {
                logger.error(err, 'S3 Upload failed');
            }).then(cacheLocally, cacheLocally) // Cache locally regardless of result
            .then(_.constant(name)); // We succeed iff cacheLocally succeeds
    } else {
        // On prod/staging ETL fails if upload fails
        logger.debug('Uploading dataset');
        return s3Upload(binData, metadata)
            .then(_.constant(name))
            .fail(function (err) {
                logger.error(err, 'S3 Upload failed');
            });
    }
}


// Buffer * {name: String, ...} -> Promise
function s3Upload(binaryBuffer, metadata) {
    return s3.upload(config.S3, config.BUCKET, metadata, binaryBuffer);
}


function parseQueryParams(req) {
    var res = [];

    res.usertag = req.query.usertag || 'unknown';
    res.agent = req.query.agent || 'unknown';
    res.agentVersion = req.query.agentversion || '0.0.0';
    res.apiVersion = parseInt(req.query.apiversion) || 0;
    res.key = req.query.key;

    return res;
}


function req2data(req, params) {
    var encoding = params.apiVersion === 0 ? 'identity'
                                           : req.headers['content-encoding'] || 'identity';

    logger.info('ETL request submitted', params);

    var chunks = [];
    var result = Q.defer();

    req.on('data', function (chunk) {
        chunks.push(chunk);
    });

    req.on('end', function () {
        var data = Buffer.concat(chunks)

        logger.debug('Request bytes:%d, encoding:%s', data.length, encoding);

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
        logger.error(err, 'ETL post fail');
        res.send({
            success: false,
            msg: err.message
        });
        logger.debug('Failed worker, exiting');
        process.exit(1);
    };
}


// Handler for ETL requests on central/etl
function jsonEtl(k, req, res) {
    var params = parseQueryParams(req);
    req2data(req, params).then(function (data) {
        try {
            etl(JSON.parse(data), params)
                .then(function (name) {
                    logger.info('ETL successful, dataset name is', name);
                    res.send({ success: true, dataset: name });
                    k();
                }, makeFailHandler(res));
        } catch (err) {
            makeFailHandler(res)(err);
        }
    }).fail(makeFailHandler(res));
}


function vgraphEtl(k, req, res) {
    req2data(req, params).then(function (data) {
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
        logger.debug('Worker finished, exiting');
        if (config.ENVIRONMENT === 'production' || config.ENVIRONMENT === 'staging') {
            process.exit(0);
        } else {
            logger.warn('not actually exiting, only disconnect socket');
            socket.disconnect();
        }
    };

    app.post('/etl', bodyParser.json({type: '*', limit: '128mb'}), jsonEtl.bind('', done));
    app.post('/etlvgraph', bodyParser.raw({type: '*', limit: '64mb'}), vgraphEtl.bind('', done))
}


module.exports = {
    route: route
};
