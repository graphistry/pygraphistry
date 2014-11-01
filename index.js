'use strict';

var path = require('path');

exports.renderer = require('./src/renderer');

exports.render_config = {
    'graph': require('./src/renderer.config.graph'),
    'superconductor': require('./src/renderer.config.sc')
};
