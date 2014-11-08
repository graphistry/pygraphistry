'use strict';

var path = require('path');

exports.renderer = require('./src/renderer');

exports.render_config = {
    'superconductor': require('./src/renderer.config.sc')
};
