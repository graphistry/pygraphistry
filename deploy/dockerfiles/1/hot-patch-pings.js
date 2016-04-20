'use strict';

var log         = require('common/logger.js');
var logger      = log.createLogger('viz-server:pings');

var config  = require('config')();

var currentlyServing = false;

function setServing(serving) {
    currentlyServing = serving;
}

function startPings(MongoClient, db) {
    // Ping home with state info
    setInterval(function(){
        db.collection('node_monitor').update(
            { 'port': config.VIZ_LISTEN_PORT, 'ip': config['HOSTNAME'], 'pid': process.pid },
            {'$set':
                { 'active': currentlyServing, 'updated': new Date() }
            },
            { 'upsert': true },
            function mongoUpdatePingCallback(err){
                if (err) {
                    logger.die(err, 'Error updating mongo, exiting');
                }
            });
    }, 3000);

}


function init (cb) {

    if(config.PINGER_ENABLED === true) {
        var MongoClient = require('mongodb').MongoClient;

        MongoClient.connect(config.MONGO_SERVER, {auto_reconnect: true}, function(err, database) {
            if(err) {
                logger.error(err, 'could not connect mongo');
                return cb(err);
            }

            try {
                var db = database.db(config.DATABASE);
                startPings(MongoClient, db);
                return cb();
            } catch (e) {
                logger.error(e, 'could not setup mongo/pings');
                return cb(e);
            }
        });

    } else {

        return cb();

    }
}

module.exports = {
    init: init,
    setServing: setServing,
    getIsServing: function () { return currentlyServing; }
};

