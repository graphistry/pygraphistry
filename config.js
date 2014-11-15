console.error("WARNING change graph-viz server.js to get config from ansible");

// WARNING: THIS FILE GETS OVER WRITTEN IN PRODUCTION.
// SEE ansible/roles/node-server/templates/config.j2

var _ = require('underscore');

module.exports = function() {
    var defaultOptions = {
        VIZ_LISTEN_ADDRESS: '0.0.0.0',
        VIZ_LISTEN_PORT: 10000,
        HTTP_LISTEN_ADDRESS: 'localhost',
        HTTP_LISTEN_PORT: 3000,
        MONGO_SERVER: 'localhost',
        DATABASE: 'graphistry-dev',
        HOSTNAME: 'localhost'
    };

    var commandLineOptions = process.argv.length > 2 ? JSON.parse(process.argv[2]) : {};

    var options = _.extend(defaultOptions, commandLineOptions);

    return options;
};
