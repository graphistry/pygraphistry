// WARNING: THIS FILE GETS OVER WRITTEN IN PRODUCTION.
// SEE ansible/roles/node-server/templates/config.j2

var _ = require('underscore');
var path = require('path');

module.exports = function() {
    var defaultOptions = {
        NODE_CL_PATH: '/opt/Superconductor2/nodecl',
        GPU_STREAMING_PATH_RELATIVE: 'GPUStreaming',
        STREAMGL_PATH_RELATIVE: 'StreamGL/src',
        LISTEN_ADDRESS: '0.0.0.0',
        LISTEN_PORT: 10000
    };

    var commandLineOptions = process.argv.length > 2 ? JSON.parse(process.argv[2]) : {};

    var options = _.extend(defaultOptions, commandLineOptions);

    options.GPU_STREAMING_PATH = path.resolve(options.NODE_CL_PATH, options.GPU_STREAMING_PATH_RELATIVE) + '/';
    options.STREAMGL_PATH = path.resolve(options.GPU_STREAMING_PATH, options.STREAMGL_PATH_RELATIVE) + '/';

    return options;
};
