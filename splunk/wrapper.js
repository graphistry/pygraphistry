var config      = require('config')();

var needle      = require('needle');
var _           = require('underscore');


if (!config.splunk) {
    throw new Error('Missing config.splunk');
}

//{scheme,host,port} -> string
function cfgToUrl(cfg) {
    return cfg.scheme + '://' + cfg.host + ':' + cfg.port + '/en-US/splunkd/__raw/servicesNS/';
}


var url = cfgToUrl(config.splunk) + 'admin/search/auth/login';
needle.post(
    url,
    _.extend({output_mode: 'json'}, config.splunk),
    {timeout: 500},
    function (err, resp) {
        if (err) {
            console.error('ERROR', err);
        } else {
            console.log('KEY', resp.body.sessionKey);
        }
    });