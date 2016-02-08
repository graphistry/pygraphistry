'use strict';

var path = require('path');
var webpack = require('webpack');

var AssetsPlugin = require('assets-webpack-plugin');
var StringReplacePlugin = require('string-replace-webpack-plugin');
var WriteFileWebpackPlugin = require('write-file-webpack-plugin');

var preLoaders = [{
    test: /\.js$/,
    loader: 'source-map-loader'
}];

var loaders = [{
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
            replacement: function(/*match, p1, offset, string*/) {
                return false;
            }
        }]
    })
}];

module.exports = {
    amd: false,
    progress: true,
    node: { fs: 'empty' },
    devtool: 'source-map',
    externals: { jquery: 'jQuery' },
    devServer: { outputPath: './dist' },
    entry: { 'StreamGL': ['./src/main.js'] },
    output: {
        // filename: '[name]-[chunkhash].js',
        // chunkFilename: '[name]-[chunkhash].js',

        path: './dist',
        publicPath: '/dist/',
        filename: '[name].js',
        chunkFilename: '[name].js',
        sourceMapFilename: '[name].map'
    },
    module: {
        loaders: loaders,
        // uncomment this when we fix Rx's sourcemaps
        // preLoaders: preLoaders
    },
    resolve: {
        modulesDirectories: ['src', 'node_modules'],
        extensions: ['', '.json', '.js', '.jsx'],
        fallback: [path.resolve('./node_modules')]
    },
    resolveLoader: {
        fallback: [path.resolve('./node_modules')]
    },
    plugins: [
        new StringReplacePlugin(),
        new webpack.NoErrorsPlugin(),
        new AssetsPlugin({ path: './dist' }),
        new webpack.optimize.OccurrenceOrderPlugin(),
        new WriteFileWebpackPlugin({ test: /(\.(js|css|map)?$)/ })
    ]
};
