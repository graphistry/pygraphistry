exports.staticFilePath = function() {
    return __dirname;
};

exports.init = require('./server-viz.js').init;