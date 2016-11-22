var chalk = require('chalk');
var webpack = require('webpack');
var Observable = require('rxjs').Observable;
var argv = process.argv.slice(2);
while (argv.length < 3) {
    argv.push(0);
}

module.exports = buildResource;

if (require.main === module) {

    var webpackConfig = require('./webpack.config.js')[argv[2]](
        process.env.NODE_ENV === 'development',
        argv[1] === '--fancy'
    );
    var isDevBuild = process.env.NODE_ENV === 'development';
    var shouldWatch = argv[0] === '--watch';
    var watcher = buildResource(
        webpackConfig, isDevBuild, shouldWatch, function(err, data) {
            if (process.send) {
                if (err) {
                    return process.send(err);
                }
                process.send(data);
                if (!shouldWatch) {
                    process.send({ type: 'complete' });
                }
            } else {
                const body = JSON.parse((err || data).body);
                if (err) {
                    console.error('%s ❌  Failed while building %s', chalk.red('[WEBPACK]'), body.name);
                    console.error(err.error);
                } else {
                    console.log('%s ✅  Successfully built %s', chalk.blue('[WEBPACK]'), body.name);
                }
            }
        }
    );

    process.on('SIGINT', function() {
        if (watcher) {
            watcher.close();
        }
        process.exit(0);
    });

    process.on('message', function(data) {
        if (data === 'die') {
            if (watcher) {
                watcher.close();
            }
            process.exit(0);
        }
    });

}

function buildResource(webpackConfig, isDevBuild, shouldWatch, cb) {

    console.log('%s %s %s', chalk.blue('[WEBPACK]'),
                 shouldWatch ? '👀  Started watching' : '🔨  Started building',
                 chalk.yellow(getAppName(webpackConfig)));

    var compiler = webpack(webpackConfig);
    var compileMethod = !shouldWatch ?
        compiler.run.bind(compiler) :
        compiler.watch.bind(compiler, {});

    return compileMethod(function(err, stats) {
        if (err || stats.hasErrors()) {

            var message = chalk.red('[WEBPACK]') + ' Errors building ' + chalk.yellow(getAppName(webpackConfig)) + "\n"
                + stats.compilation.errors.map(function(error) {
                    return error.message;
                }).join("\n");

            console.error('%s fatal error occured', chalk.red('[WEBPACK]'));
            // console.error(err);
            if (shouldWatch) {
                console.log(message);
                // console.log(stats.toString(webpackConfig.stats || {}));
            } else {
                return cb({
                    type: 'error',
                    error: message,
                    body: JSON.stringify({
                        name: getAppName(webpackConfig),
                        stats: stats.toString(webpackConfig.stats || {})
                    })
                });
            }
        }
        cb(null, { type: 'next', body: JSON.stringify({
            name: getAppName(webpackConfig),
            stats: stats.toJson(webpackConfig.stats || { entrypoints: true })
        })});
    });

    function getAppName(webpackConfig) {
        var appName = webpackConfig.name || webpackConfig.output.filename;
        if(~appName.indexOf('[name]') && typeof webpackConfig.entry === 'object') {
            var entryNames = Object.keys(webpackConfig.entry);
            if(entryNames.length === 1) {
                // we can only replace [name] with the entry point if there is only one entry point
                appName = appName.replace(/\[name]/, entryNames[0]);
            }
        }
        return appName;
    }
}
