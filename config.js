// WARNING: THIS FILE GETS OVER WRITTEN IN PRODUCTION.
// SEE ansible/roles/node-server/templates/config.j2

var config = {}

config.NODE_CL_PATH = "/opt/Superconductor2/nodecl/"
config.GPU_STREAMING_PATH = config.NODE_CL_PATH + "GPUStreaming/"
config.STREAMGL_PATH = config.GPU_STREAMING_PATH + "StreamGL/src/";

// Default IP and port the server listens on. Can be overridden by the user by passing an argument
// to this script on the command line of form <IP>:<PORT>. <IP> is either 4 numbers ('192.169.0.1')
// or 'localhost'; <PORT> is a number. Both are optional. If only 1 is supplied, ':' is optional.
config.DEFAULT_LISTEN_ADDRESS = 'localhost';
config.DEFAULT_LISTEN_PORT = 10000;

module.exports = config;