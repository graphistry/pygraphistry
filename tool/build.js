var http = require('http');
var path = require('path');
var chalk = require('chalk');
var Subject = require('rxjs').Subject;
var Observable = require('rxjs').Observable;
var Subscription = require('rxjs').Subscription;
var ReplaySubject = require('rxjs').ReplaySubject;
var childProcess = require('child_process');
var buildResource = require('./build-resource');
var webpackConfigs = require('./webpack.config.js');

var HMRPort = 8090;
var HMRMiddleware = require('webpack-hot-middleware');
var isFancyBuild = process.argv[3] === '--fancy';
var isDevBuild = process.env.NODE_ENV === 'development';
var shouldWatch = isDevBuild && process.argv[2] === '--watch';

var clientConfig = webpackConfigs[0](isDevBuild, isFancyBuild);
var serverConfig = webpackConfigs[1](isDevBuild, isFancyBuild);

// copy static assets
var shelljs = require('shelljs');
shelljs.mkdir('-p', clientConfig.output.path);
shelljs.cp('-rf', './src/viz-client/static/*', clientConfig.output.path);
shelljs.mkdir('-p', serverConfig.output.path);
shelljs.cp('-rf', './src/viz-server/static/*', serverConfig.output.path);
shelljs.cp('-rf', './src/viz-worker/static/*', serverConfig.output.path);

var compile, compileClient, compileServer;

// Prod builds have to be built sequentially so webpack can share the css modules
// style cache between both client and server compilations.
if (!isDevBuild) {
    compileClient = buildResourceToObservable(clientConfig, isDevBuild, shouldWatch);
    compileServer = buildResourceToObservable(serverConfig, isDevBuild, shouldWatch);
    compile = Observable.concat(compileClient, compileServer).take(2).toArray();
}
// Dev builds can run in parallel because we don't extract the client-side
// CSS into a styles.css file (which allows us to hot-reload CSS in dev mode).
else {
    compileClient = processToObservable(childProcess
        .fork(require.resolve('./build-resource'), process.argv.slice(2).concat([0, 0]), {
            env: process.env, cwd: process.cwd()
        }));
    compileServer = processToObservable(childProcess
        .fork(require.resolve('./build-resource'), process.argv.slice(2).concat([1, 1]), {
            env: process.env, cwd: process.cwd()
        }));
    compile = Observable.combineLatest(compileClient, compileServer);
}

compile.do({
        next: function(res) {
            console.log('%s ✅  Build succeeded', chalk.blue('[WEBPACK]'));
        },
        error(err) {
            console.error('%s ❌  Build failed', chalk.blue('[WEBPACK]'));
            console.error(err.toString());
        }
    })
    .multicast(function() { return new Subject(); }, function(shared) {

        if (!shouldWatch) {
            return shared;
        }

        return Observable.merge(
            shared.take(1).mergeMap(createClientServer),
            shared.map(function(arr) { return arr[1] })
                .distinctUntilChanged()
                .mergeScan(startOrUpdateServer, null)
        );

        function createClientServer() {
            console.log('Starting Client [HMR] Server...');
            var clientHMRServer = new http.createServer(function(req, res) {
                res.setHeader('Access-Control-Allow-Origin', '*');
                HMRMiddleware({
                    plugin: function(type, cb) {
                        if (type === 'done') {
                            shared
                                .map(function(arr) { return arr[0]; })
                                .distinctUntilChanged()
                                .subscribe(cb);
                        }
                    }
                }, { log: false })(req, res);
            });
            var listenAsObs = Observable.bindNodeCallback(clientHMRServer.listen.bind(clientHMRServer), function() {
                console.log('************************************************************');
                console.log('Client HMR server listening at http://%s:%s', this.address().address, this.address().port);
                console.log('************************************************************');
                return clientHMRServer;
            })
            return listenAsObs(HMRPort);
        }

        function startOrUpdateServer(child, stats) {
            if (!child) {
                console.log('Starting Dev Server with [HMR]...');
            } else {
                child.kill('SIGKILL');
                console.log('Restarting Dev Server with [HMR]...');
            }
            child = childProcess.fork(path.join(
                serverConfig.output.path,
                serverConfig.output.filename), {
                env: process.env, cwd: process.cwd()
            });
            return processToObservable(child)
                .last(null, null, { code: 1 })
                .mergeMap(function(data) {
                    if (data && data.code != null) {
                        console.error(
                            'Dev Server exited with code:', data.code,
                            'and signal:', data.signal
                        );
                    }
                    return Observable.empty();
                })
                .startWith(child);
        }
    })
    .subscribe({
        error: function(e) {
            console.error('Unhandled compilation error: ', e);
        }
    });

process.on('exit', function() {
    require('tree-kill')(process.pid, 'SIGKILL');
});

function buildResourceToObservable(webpackConfig, isDevBuild, shouldWatch) {
    var subject = new ReplaySubject(1);
    return Observable.using(function() {
        var watcher = buildResource(webpackConfig, isDevBuild, shouldWatch, function(err, data) {
            if (err) {
                return subject.error(err);
            }
            subject.next(JSON.parse(data.stats));
            if (!shouldWatch) {
                subject.complete();
            }
        });
        return new Subscription(function() {
            if (watcher) {
                watcher.close();
            }
        });
    }, function(subscription) {
        return subject;
    });
}

function processToObservable(process) {
    return Observable.create(function(subscriber) {
        function onExitHandler(code, signal) {
            if (code != null) {
                subscriber.next({
                    code: code,
                    signal: signal
                });
            }
            subscriber.complete();
        }
        function onMessageHandler(data) {
            if (!data) {
                return;
            } else if (data.type === 'complete') {
                return subscriber.complete();
            } else if (data.type === 'next') {
                return subscriber.next(JSON.parse(data.stats));
            } else if (data.type === 'error') {
                subscriber.error({
                    error: data.error,
                    stats: JSON.parse(data.stats)
                });
            }
        };
        process.on('exit', onExitHandler);
        process.on('message', onMessageHandler);
        return function() {
            // if (process.exitCode === null) {
                process.kill('SIGKILL');
            // }
        }
    });
}
