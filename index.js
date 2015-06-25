var Log         = require('common/logger.js');
var logger      = Log.createLogger('etlworker:index');
var etl     = require('./etl.js');

exports.staticFilePath = function() {
    return __dirname;
};

exports.init = function init(app, socket) {
    logger.debug('Client connected', socket.id);

    etl.route(app, socket);
};
