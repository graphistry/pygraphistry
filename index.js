var debug   = require('debug')('graphistry:etlworker:index');
var bodyParser  = require('body-parser');

var etl     = require('./etl.js');

exports.staticFilePath = function() {
    return __dirname;
};

exports.init = function init(app, socket) {
    debug('Client connected', socket.id);

    // Temporarly handle ETL request from Splunk
    app.post('/etl', bodyParser.json({type: '*', limit: '64mb'}), etl.post);

    return {
        getState: function () {
            return {
                then: function () {
                    return {
                        done: function (f) { return f(); }
                    };
                }
            };
        }
    };
};