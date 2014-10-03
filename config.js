// WARNING: THIS FILE GETS OVER WRITTEN IN PRODUCTION.
// SEE ansible/roles/node-server/templates/config.j2

var _ = require('underscore');

function config (overrides) {

    var NODE_CL_PATH = overrides.NODE_CL_PATH || "/opt/Superconductor2/nodecl/";
    var GPU_STREAMING_PATH = NODE_CL_PATH + "GPUStreaming/";
    var STREAMGL_PATH = GPU_STREAMING_PATH + "StreamGL/src/";

    return {

        NODE_CL_PATH: NODE_CL_PATH,
        GPU_STREAMING_PATH: GPU_STREAMING_PATH,
        STREAMGL_PATH: STREAMGL_PATH,

        // Default IP and port the server listens on. Can be overridden by the user by passing an argument
        // to this script on the command line of form <IP>:<PORT>. <IP> is either 4 numbers ('192.169.0.1')
        // or 'localhost'; <PORT> is a number. Both are optional. If only 1 is supplied, ':' is optional.
        DEFAULT_LISTEN_ADDRESS: 'localhost',
        DEFAULT_LISTEN_PORT: 10000
    };
}

module.exports = function (overrides) {
    overrides = overrides || {};
    return _.extend(config(overrides), overrides);
};
