var webpack = require('webpack');
var config = require('./webpack.config');
var ExtractTextPlugin = require('extract-text-webpack-plugin');

// config.devtool = 'eval-cheap-module-source-map';
config.module.loaders.push(
    { test: /\.(c|le)ss$/, loader: 'style!css?modules&importLoaders=2&sourceMap&localIdentName=[local]___[hash:base64:5]!postcss' }
);

config.plugins.push(
    new webpack.DefinePlugin({
        __DEV__: true,
        'process.env.NODE_ENV': '"development"',
    }),
    new ExtractTextPlugin('styles.css', { allChunks: true })
);

module.exports = config;
