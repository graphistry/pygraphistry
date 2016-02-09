var webpack = require('webpack');
var StringReplacePlugin = require('string-replace-webpack-plugin');

module.exports = function(config) {

    config.set({

        basePath: '..',

        browsers: ['PhantomJS'],

        // singleRun: !!process.env.CONTINUOUS_INTEGRATION,
        singleRun: true,

        // frameworks: ['mocha'],
        frameworks: ['jasmine'],

        files: [
            'spec/*Spec.js'
        ],

        preprocessors: {
            'spec/*Spec.js': ['webpack', 'sourcemap']
        },

        reporters: ['spec'],

        plugins: [
            require("karma-webpack"),
            require("karma-jasmine"),
            require("karma-spec-reporter"),
            require("karma-phantomjs-launcher"),
            require("karma-sourcemap-loader")
        ],

        webpack: {
            devtool: 'inline-source-map',
            module: {
                loaders: [{
                    test: /\.jsx?$/,
                    exclude: /(node_modules)/,
                    loader: 'babel',
                    query: {
                        // presets: ['react', 'es2015', 'stage-0'],
                        // plugins: ['transform-runtime'], //
                        // Use `require.resolve` here because we require locally linked modules that
                        // don't have these plugins in their node_modules tree.
                        presets: [
                            require.resolve('babel-preset-react'),
                            require.resolve('babel-preset-es2015'),
                            require.resolve('babel-preset-stage-0')
                        ],
                        plugins: [
                            require.resolve('babel-plugin-transform-runtime')
                        ],
                        cacheDirectory: true // cache into OS temp folder by default
                    }
                },
                { test: /\.(png|jpg|jpeg|gif|mp3)$/, loader: 'url?limit=10000&name=[name]_[hash:6].[ext]' },
                { test: /\.txt$/, loader: 'raw' },
                { test: /\.json$/, loader: 'json' },
                { test: /\.pegjs$/, loader: 'pegjs-loader?cache=true&optimize=size' },
                {
                    test: /PEGUtil.js$/,
                    include: /node_modules\/pegjs-util/,
                    loader: StringReplacePlugin.replace({ // from the 'string-replace-webpack-plugin'
                        replacements: [{
                            pattern: /typeof define\.amd !== (\"|\')undefined(\"|\')/ig,
                            replacement: function() {
                                return false;
                            }
                        }]
                    })
                }]
            },
            resolve: {
                modulesDirectories: [
                    'src',
                    'node_modules'
                ],
                extensions: ['', '.json', '.js']
            },
            plugins: [
                new StringReplacePlugin(),
                new webpack.IgnorePlugin(/\.json$/),
                new webpack.NoErrorsPlugin(),
                new webpack.DefinePlugin({
                    __DEV__: true,
                    'process.env.NODE_ENV': '"development"',
                })
            ]
        },
        webpackServer: {
            noInfo: true
        }
    });
};
