'use strict';

var VizServer = require('./server-viz.js');

VizServer.staticFilePath = function() {
    return __dirname;
};

module.exports = VizServer;
