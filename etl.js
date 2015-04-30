var urllib   = require('url');
var zlib     = require('zlib');
var debug    = require('debug')('graphistry:etlworker:etl');
var _        = require('underscore');
var Q        = require('q');
var bodyParser  = require('body-parser');

var config   = require('config')();

var vgraph   = require('./vgraph.js');
var Cache = require('common/cache.js');

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
    var metadata = {name: name};
    var binData = vg.encode().toBuffer();

    function cacheLocally() {
        // Wait a couple of seconds to make sure our cache has a
        // more recent timestamp than S3
        var res = Q.defer();
        setTimeout(function () {
            debug('Caching dataset locally');
            res.resolve(tmpCache.put(urllib.parse(name)), binData);
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
    debug('uploading VGraph', metadata.name);

    return Q.nfcall(zlib.gzip, binaryBuffer)
        .then(function (zipped) {
            var params = {
                Bucket: config.BUCKET,
                Key: metadata.name,
                ACL: 'private',
                Metadata: metadata,
                Body: zipped,
                ServerSideEncryption: 'AES256'
            };

            debug('Upload size', (zipped.length/1000).toFixed(1), 'KB');
            return Q.nfcall(config.S3.putObject.bind(config.S3), params);
        })
        .then(function () {
            debug('Upload done', metadata.name);
        });
}


// Handler for ETL requests on central/etl
function post(k, req, res) {
    var data = "";

    req.on('data', function (chunk) {
        data += chunk;
    });

    req.on('end', function () {
        var fail = function (err) {
            console.error('ETL post fail', (err||{}).stack);
            res.send({
                success: false,
                msg: JSON.stringify(err)
            });
            console.error('Failed worker, exiting');
            process.exit(1);
        };

        try {
            etl(JSON.parse(data))
                .done(
                    function (name) {
                        debug('ETL done, notifying client to proceed');
                        //debug('msg', msg);
                        res.send({ success: true, dataset: name });
                        debug('notified');
                        k();
                    }, fail);
        } catch (err) {
            fail(err);
        }
    });
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

    // Temporarly handle ETL request from Splunk
    app.post('/etl', bodyParser.json({type: '*', limit: '64mb'}), post.bind('', done));

}



module.exports = {
    post: post,
    route: route
}
