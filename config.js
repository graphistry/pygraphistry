// WARNING: THIS FILE GETS OVER WRITTEN IN PRODUCTION.
// SEE ansible/roles/node-server/templates/config.j2

var _ = require('underscore');

function config (overrides) {

    var NODE_CL_PATH = overrides.NODE_CL_PATH || "/opt/Superconductor2/nodecl/";
    var GPU_STREAMING_PATH = NODE_CL_PATH + "GPUStreaming/";
    var STREAMGL_PATH = GPU_STREAMING_PATH + "StreamGL/src/";
    var LISTEN_ADDRESS = overrides.LISTEN_ADDRESS || '0.0.0.0';
    var LISTEN_PORT = overrides.LISTEN_PORT || 10000;

    return {

        NODE_CL_PATH: NODE_CL_PATH,
        GPU_STREAMING_PATH: GPU_STREAMING_PATH,
        STREAMGL_PATH: STREAMGL_PATH,
        LISTEN_ADDRESS: LISTEN_ADDRESS,
        LISTEN_PORT: LISTEN_PORT
    };
}

module.exports = function (overrides) {
    overrides = overrides || {};
    return _.extend(config(overrides), overrides);
};
