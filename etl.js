var urllib   = require('url');

var debug    = require('debug')('graphistry:etlworker:etl');
var _        = require('underscore');
var Q        = require('q');

var config   = require('config')();

var vgraph   = require('./vgraph.js');
var vgwriter = require('../graph-viz/js/libs/VGraphWriter.js');
var loader = require('../graph-viz/js/data-loader.js');


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
    }

    var metadata = {name: name};

    function cacheLocally() {
        // Wait a couple of seconds to make sure our cache has a
        // more recent timestamp than S3
        var res = Q.defer();
        setTimeout(function () {
            debug('Caching dataset locally');
            res.resolve(loader.cache(vg.encode().toBuffer(), urllib.parse(name)));
        }, 2000);
        return res.promise;
    }

    if (config.ENVIRONMENT === 'local') {
        debug('Attempting to upload dataset');
        return vgwriter.uploadVGraph(vg, metadata)
            .fail(function (err) {
                console.error('S3 Upload failed', err.message);
            }).then(cacheLocally, cacheLocally) // Cache locally regardless of result
            .then(_.constant(name)); // We succeed iff cacheLocally succeeds
    } else {
        // On prod/staging ETL fails if upload fails
        debug('Uploading dataset')
        return vgwriter.uploadVGraph(vg, metadata)
            .then(_.constant(name))
            .fail(function (err) {
                console.error('S3 Upload failed', err.message);
            });
    }
}

// Handler for ETL requests on central/etl
function post(req, res) {
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
                        debug('worker finished, exiting');
                        if (config.ENVIRONMENT === 'production' || config.ENVIRONMENT === 'staging') {
                            process.exit(0);
                        } else {
                            console.warn('not actually exiting');
                        }
                    }, fail);
        } catch (err) {
            fail(err);
        }
    });
}

module.exports = {
    post: post
}
