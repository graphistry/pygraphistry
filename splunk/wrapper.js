/*
    Helper methods for accessing splunk REST APIs
    (Replacement for Splunk Node SDK, which was failing)

    Usage: ./cmd.sh

    To use: create config.json (passed in via cmd.sh)

        {"splunk": {
            "username", "password",
            "scheme": "https",
            "host",
            "port": 8089,
            "version": "0.2",
            "prefix": "/servicesNS"
        }}


    1. Get auth token. Lasts ~1 hr since last API call.
    2. Make an async search job with it
    3. Poll on that job till completion..

    Example:

        Get result of query "source=stream:http site != splunk.graphistry.com:3000"

            search('source=stream:*  | stats count(bytes_in), count(bytes_out), min(timestamp), max(timestamp) by dest_ip, dest_port, src_ip')
                .takeLast(1)
                .subscribe(function () { console.log('SUCCEED'); }, console.error.bind(console, 'FAILED'));

        Include intermediate results:

            search('source=stream:*  | stats count(bytes_in), count(bytes_out), min(timestamp), max(timestamp) by dest_ip, dest_port, src_ip')
                .subscribe(function () { console.log('SUCCEED'); }, console.error.bind(console, 'FAILED'));



*/

var config      = require('config')();

var debug       = require('debug')('graphistry:graph-viz:splunk:wrapper');
var needle      = require('needle');
var _           = require('underscore');
var Rx          = require('rx');


var needleRx = {};
for (var i in needle) {
    if (typeof(needle[i]) === 'function') {
        needleRx[i] = Rx.Observable.fromNodeCallback(needle[i].bind(needle));
    }
}

var BASE_SPLUNK_OPTIONS = {timeout: 500, rejectUnauthorized: false};


if (!config.splunk) {
    throw new Error('need config.splunk');
}

//{scheme,host,port,?prefix,username} -> string
function cfgToUrl(cfg) {
    return cfg.scheme + '://' + cfg.host + ':' + cfg.port
        + (cfg.prefix || '')
        + '/' + cfg.username + '/'
        + (cfg.postfix || '');
}

//{scheme,host,port,password,url} -> Observable string
function getSessionKey(cfg) {

    var url = cfgToUrl(cfg) + 'search/auth/login';

    return Rx.Observable.return()
        .flatMap(function () { //wrap to enable retry
            debug('Request session:', url);
            return needleRx.post(
                url,
                {output_mode: 'json', username: cfg.username, password: cfg.password},
                BASE_SPLUNK_OPTIONS) //ignore self-signed cert
        })
        .pluck('0').pluck('body').pluck('sessionKey')
        .do(function (key) { debug('  -> received session key', key); })
        .retryWhen(function (errors) {
            console.warn('Auth error, wait 50ms and retry');
            return errors.delay(50);
        });

}



//{scheme,host,port} * String * String -> Observable string
function makeSearchJob (cfg, sessionKey, str) {
    var url = cfgToUrl(cfg) + 'search/search/jobs';
    debug('Request create job:', url, str);
    return needleRx.post(
            url,
            {search: 'search ' + str, output_mode: 'json'},
            _.extend({}, BASE_SPLUNK_OPTIONS, {headers: {Authorization: 'Splunk ' + sessionKey}}))
        .pluck('0').pluck('body').pluck('sid')
        .do(function (result) { debug('  -> received new job: ', result); })
}

//{scheme,host,port} * String * String -> Observable string
//Poll until done
function pollSearch(cfg, sessionKey, sid) {

    var url = cfgToUrl(cfg) + 'search/search/jobs/' + sid + '/events?output_mode=json';
    debug('pollSearch', url);

    var done = false;

    return Rx.Observable.return()
    .expand(function () {
        return needleRx.get(
            url,
            _.extend({}, BASE_SPLUNK_OPTIONS, {headers: {Authorization: 'Splunk ' + sessionKey}}))
    })
    .filter(_.identity).pluck('0').pluck('body')
    .takeWhile(function (o) { return !done; })
    .do(function (result) {
        done = !result.preview;
        debug('   -> streaming results', _.extend({}, result, {results: _.range(0, result.results.length)}));
    });
}


/*

    Get (streaming) matches for a search

    string ->
        Observable {
            preview: bool, //true if intermediate
            fields: [ {name: string }],
            results: [ {key->val} ]
        }
*/
function search (query) {

    return getSessionKey(config.splunk)
    .flatMap(function (key) {
        return makeSearchJob(config.splunk, key, query)
        .flatMap(pollSearch.bind('', config.splunk, key))
    })

}


module.exports = {
    search: search
};

/*
search('source=stream:*  | stats count(bytes_in), count(bytes_out), min(timestamp), max(timestamp) by dest_ip, dest_port, src_ip')
    .takeLast(3)
    .subscribe(console.log.bind('', 'SUCCEED'), console.error.bind(console, 'FAILED'));
*/