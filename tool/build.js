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

var pid = process.pid;
var argv = process.argv.slice(2);
while (argv.length < 2) {
    argv.push(0);
}

var HMRPort = 8090;
var HMRMiddleware = require('webpack-hot-middleware');
var shouldWatch = argv[0] === '--watch';
var isFancyBuild = argv[1] === '--fancy';
var isDevBuild = process.env.NODE_ENV === 'development';

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

// Dev builds can run in parallel because we don't extract the client-side
// CSS into a styles.css file (which allows us to hot-reload CSS in dev mode).
if (isDevBuild) {

    compileClient = processToObservable(childProcess
        .fork(require.resolve('./build-resource'), argv.concat(0), {
            env: process.env, cwd: process.cwd()
        }));

    compileServer = processToObservable(childProcess
        .fork(require.resolve('./build-resource'), argv.concat(1), {
            env: process.env, cwd: process.cwd()
        }));

    compile = Observable.combineLatest(compileClient, compileServer);
}
// Prod builds have to be built sequentially so webpack can share the css modules
// style cache between both client and server compilations.
else {

    compileClient = buildResourceToObservable(
        clientConfig, isDevBuild, shouldWatch
    ).multicast(() => new Subject()).refCount();

    compileServer = buildResourceToObservable(
        serverConfig, isDevBuild, shouldWatch
    ).multicast(() => new Subject()).refCount();

    compile = compileClient.mergeMap(
            (client) => compileServer,
            (client, server) => [client, server]
        )
        .take(1)
        .mergeMap((results) => !shouldWatch ?
            Observable.of(results) :
            Observable.combineLatest(
                compileClient.startWith(results[0]),
                compileServer.startWith(results[1])
            )
        );
}

compile.multicast(function() { return new Subject(); }, function(shared) {

        const client = shared.map((xs) => xs[0]).distinctUntilChanged();
        const server = shared.map((xs) => xs[1]).distinctUntilChanged();

        const buildStatuses = client.merge(server);
        const initialBuilds = client.take(1).merge(server.take(1));

        if (!shouldWatch) {
            return buildStatuses.do({
                next: function({ name }) {
                    console.log('%s ✅  Successfully built %s', chalk.blue('[WEBPACK]'), name);
                }
            });
        }

        return Observable.merge(
            initialBuilds.do({
                next: function({ name }) {
                    console.log('%s ✅  Successfully built %s', chalk.blue('[WEBPACK]'), name);
                }
            }),
            buildStatuses.skipUntil(initialBuilds).do({
                next: function({ name }) {
                    console.log('%s ✅  Successfully rebuilt %s', chalk.blue('[WEBPACK]'), name);
                }
            }),
            shared.take(1).mergeMap(createClientServer),
            server.mergeScan(startOrUpdateServer, null)
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
        error(err) {
            console.error('%s ❌  Failed while building', chalk.red('[WEBPACK]'));
            console.error(err.error);
            console.error(err.stats);
        }
    });

process.on('exit', function() {
    require('tree-kill')(pid, 'SIGKILL');
});

function buildResourceToObservable(webpackConfig, isDevBuild, shouldWatch) {
    var subject = new ReplaySubject(1);
    return Observable.using(function() {
        var watcher = buildResource(webpackConfig, isDevBuild, shouldWatch, function(err, data) {
            if (err) {
                return subject.error({
                    error: err.error,
                    stats: err.stats
                });
            }
            subject.next(JSON.parse(data.body));
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
                return subscriber.next(JSON.parse(data.body));
            } else if (data.type === 'error') {
                subscriber.error({
                    error: data.error,
                    stats: data.stats
                });
            }
        };
        process.on('exit', onExitHandler);
        process.on('message', onMessageHandler);
        return function() {
            process.kill('SIGKILL');
        }
    });
}
