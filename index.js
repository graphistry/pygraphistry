'use strict';

var path = require('path');

exports.renderer = require('./src/renderer');

exports.render_config = {
    'graph': require('./src/renderer.config.graph'),
    'superconductor': require('./src/renderer.config.sc')
};

// FIXME: Absolute path to static files (HTML/CSS/JS) that should be served by the HTTP server
// (This is a temporary hack. Should point to the `superconductor2/nodecl/GPUSteaming` directory.)
exports.STATIC_HTTP_PATH = path.resolve(__dirname, '../');
