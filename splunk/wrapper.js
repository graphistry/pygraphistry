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

            search('source=stream:*  | stats sum(bytes_in), sum(bytes_out), min(timestamp), max(timestamp) by dest_ip, dest_port, src_ip')
                .takeLast(1)
                .subscribe(function () { console.log('SUCCEED'); }, console.error.bind(console, 'FAILED'));

        Include intermediate results:

            search('source=stream:*  | stats sum(bytes_in), sum(bytes_out), min(timestamp), max(timestamp) by dest_ip, dest_port, src_ip')
                .subscribe(function () { console.log('SUCCEED'); }, console.error.bind(console, 'FAILED'));



*/

'use strict';

var config      = require('config')();
var debug       = require('debug')('graphistry:graph-viz:splunk:wrapper');
var needle      = require('needle');
var _           = require('underscore');
var Rx          = require('rx');
var vgraph      = require('./vgraph.js')
var vgwriter    = require('../js/libs/VGraphWriter.js');


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

function hoptions(sessionKey) {
    return _.extend({}, BASE_SPLUNK_OPTIONS,
                    {headers: {Authorization: 'Splunk ' + sessionKey}});
}

//{scheme,host,port} * String * String -> Observable string
function makeSearchJob (cfg, sessionKey, str) {
    var url = cfgToUrl(cfg) + 'search/search/jobs';
    debug('Request create job:', url, str);
    var params = {search: 'search ' + str, status_buckets: 300, output_mode: 'json'}
    return needleRx.post(url, params, hoptions(sessionKey))
        .pluck('0').pluck('body').pluck('sid')
        .do(function (result) { debug('  -> received new job: ', result); })
}

//{scheme,host,port} * String * String -> Observable string
//Poll until done
function pollSearch(cfg, sessionKey, sid) {

    var url = cfgToUrl(cfg) + 'search/search/jobs/' + sid + '/events?output_mode=json&count=0';
    debug('pollSearch', url);

    var replies = Rx.Observable.return()
    .expand(function () {
        return needleRx.get(url, hoptions(sessionKey));
    })
    .filter(_.identity).pluck('0').pluck('body');

    var isDone = function(r) { return !r.preview; };

    return Rx.Observable.merge(
        replies.takeUntil(replies.filter(isDone)),
        replies.filter(isDone).take(1)
    ).do(function (result) {
        debug('   -> streaming results', _.extend({}, result, {results: _.range(0, result.results.length)}));
    });
}

// Poll job status until done.
function pollStatus(cfg, sessionKey, sid) {
    var url = cfgToUrl(cfg) + 'search/search/jobs?output_mode=json&count=1&sid=' + sid;
    debug('pollStatus', url);

    return Rx.Observable.interval(50)
    .flatMap(function () {
        return needleRx.get(url, hoptions(sessionKey));
    })
    .pluck('0').pluck('body').pluck('entry').pluck('0').pluck('content')
    .filter(function (status) {
        console.log('Search status:%s\tprogress:%d%%',
                    status.dispatchState, (status.doneProgress * 100).toFixed(0));
        return status.isDone
    }).take(1).do(function(status) {
        console.log('Search done. Number of results', status.resultCount);
    }).map(function () {
        return sid;
    });
}

// Fetch job results
function getResults(cfg, sessionKey, sid) {
    var url = cfgToUrl(cfg) + 'search/search/jobs/' + sid + '/results?output_mode=json&count=100';
    debug('getResults', url);

    var options = _.extend(BASE_SPLUNK_OPTIONS, {headers: {Authorization: 'Splunk ' + sessionKey}});
    return needleRx.get(url, options)
    .pluck('0').pluck('body').do(function (r) {
        debug('getResults reply', r);
    });
}

// Delete job (necessary to avoid splunk license limits)
function deleteJob(cfg, sessionKey, sid) {
    var url = cfgToUrl(cfg) + 'search/search/jobs/' + sid + '/control?action=cancel&output_mode=json';
    debug('deleteJob', url);

    var params = {action: 'cancel', output_mode: 'json'};
    needleRx.post(url, params, hoptions(sessionKey))
    .pluck('0').pluck('body').do(function (r) {
        debug('deleteJob reply', r);
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
function search(query) {
    return getSessionKey(config.splunk)
    .flatMap(function (key) {
        return makeSearchJob(config.splunk, key, query)
        .flatMap(function (sid) {
            return pollStatus(config.splunk, key, sid)
            .flatMap(getResults.bind('', config.splunk, key))
            .do(function () {
                deleteJob(config.splunk, key, sid)
            });
        });
    })
}

// Apply fn onto the results of a seach query
function process(reply, fn) {
    reply.subscribe(function (res) {
        if (res.preview) {
            console.log('Warning: results are incomplete');
        }
        fn(res.results);
    }, function (err) {
        console.error('Error', err, (err || {}).stack);
    });
}

var query = 'source=stream:*  | stats sum(bytes_in) as edgeWeight, sum(bytes_out) as edgeColor, min(timestamp) as timeAppear, max(timestamp) as timeDisappear by dest_ip, dest_port, src_ip';

var metadata = {
    name: 'SplunkTest',
    type: 'vgraph',
    config: {
        simControls: 'netflow',
        scene: 'netflow',
        mapper: 'debugMapper'
    }
};


process(search(query), function (results) {
    var vg = vgraph.fromEdgeList(results, metadata.name);

    vgwriter.uploadVGraph(vg, metadata).then(function () {
        console.log('VGraph uploaded as', metadata.name);
    }).fail(function (err) {
        console.error('Error uploading vgraph', err, (err || {}).stack);
    });
});


module.exports = {
    search: search
};
