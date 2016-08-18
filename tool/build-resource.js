var chalk = require('chalk');
var webpack = require('webpack');
var Observable = require('rxjs').Observable;

module.exports = buildResource;

if (require.main === module) {

    var webpackConfig = require('./webpack.config.js')[process.argv[4]](
        process.env.NODE_ENV === 'development',
        process.argv[3] === '--fancy'
    );
    var isDevBuild = process.env.NODE_ENV === 'development';
    var shouldWatch = process.argv[2] === '--watch';
    var watcher = buildResource(
        webpackConfig, isDevBuild, shouldWatch, function(err, data) {
            if (err) {
                return process.send(err);
            }
            process.send(data);
            if (!shouldWatch) {
                process.send({ type: 'complete' });
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

    console.log('%s Started %s %s', chalk.blue('[WEBPACK]'),
                 shouldWatch ? 'watching' : 'building',
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
            console.error(err);
            if (shouldWatch) {
                console.log(message);
            } else {
                return cb({
                    type: 'error',
                    error: message,
                    stats: JSON.stringify(
                        stats.toJson(webpackConfig.stats || {}), null, 2
                    )
                });
            }
        }
        cb(null, { type: 'next', stats: JSON.stringify(
            stats.toJson(webpackConfig.stats || {}), null, 2
        )});
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
