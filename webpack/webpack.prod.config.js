var webpack = require('webpack');
var config = require('./webpack.config');
var ExtractTextPlugin = require('extract-text-webpack-plugin');

config.module.loaders.push(
    { test: /\.(c|le)ss$/, loader: ExtractTextPlugin.extract('style', 'css?modules&importLoaders=2&sourceMap!postcss') }
);

config.plugins.push(
    new webpack.DefinePlugin({
        __DEV__: false,
        'process.env.NODE_ENV': '"production"',
    }),
    new ExtractTextPlugin('styles.css', {
        allChunks: true,
    }),
    new webpack.optimize.DedupePlugin(),
    new webpack.optimize.AggressiveMergingPlugin(),
    new webpack.optimize.UglifyJsPlugin({
        compress: {
            warnings: false,
        },
        // sourceMap: false,
        comments: false,
        'screw-ie8': true,
    })
);

module.exports = config;
