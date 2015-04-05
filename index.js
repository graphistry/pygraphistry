var debug   = require('debug')('graphistry:etlworker:index');

var etl     = require('./etl.js');

exports.staticFilePath = function() {
    return __dirname;
};

exports.init = function init(app, socket) {
    debug('Client connected', socket.id);

    etl.route(app, socket);

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