'use strict' // eslint-disable-line

const webpack = require('webpack');
const config = require('@graphistry/config')();
const vizAppPackage = require('../package.json');
const AssetsPlugin = require('assets-webpack-plugin')
const StringReplacePlugin = require('string-replace-webpack-plugin');
const postcss = () => [require('postcss-calc'), require('postcss-nesting'), require('postcss-css-variables'), require('autoprefixer')]

const commonPlugins = [
  new StringReplacePlugin(),
  new webpack.optimize.OccurrenceOrderPlugin(),
  new webpack.NoErrorsPlugin(),
  new webpack.ProvidePlugin({
    h: 'snabbdom/h'
  })
]

const clientConfig = {
  amd: false,
  progress: true,
  devtool: 'source-map',
  node: { fs: 'empty' },
  entry: {
    client: ['./src/viz-client/index.js'],
  },
  output: {
    path: './www',
    publicPath: '/graph/',
    filename: '[name].js',
    //during development
    // filename: '[name]_[hash:6].js',
  },
  module: {
    loaders: [...commonLoadersWithPresets(['es2015', 'stage-0', 'react'])],
    noParse: [],
  },
  postcss,
  plugins: [...commonPlugins,
    new AssetsPlugin({
      path: './www',
    }),
    new webpack.DefinePlugin({
      __CLIENT__: true,
      __SERVER__: false,
      __RELEASE__: JSON.stringify(config.RELEASE),
      __VERSION__: JSON.stringify(vizAppPackage.version)
    }),
    /*
    // extract common code to a single file loaded asynchronously
    new webpack.optimize.CommonsChunkPlugin({
      // (the commons chunk name)
      // name: "commons",
      // (the filename of the commons chunk)
      // filename: "commons.js",
      // (Modules must be shared between at least 2 entries)
      minChunks: 2,
      // (use all children of the chunk)
      children: true,
      // (create an async commons chunk)
      async: true
    }),
    */
  ],
  externals: [
    { jquery: 'jQuery' } // externalize jQuery
  ]
}

const serverConfig = {
  devtool: 'source-map',
  entry: {
    server: ['./src/viz-server/index.js'],
  },
  target: 'node',
  output: {
    path: './www',
    filename: 'viz-server.js',
    libraryTarget: 'commonjs2',
  },
  module: {
    loaders: [...commonLoadersWithPresets(['es2015', 'stage-0', 'react'])], // can use node5 instead of es2015 when uglify-js can handle es6
  },
  postcss,
  plugins: [...commonPlugins,
    new webpack.DefinePlugin({
      __CLIENT__: false,
      __SERVER__: true,
      __RELEASE__: JSON.stringify(config.RELEASE),
      __VERSION__: JSON.stringify(vizAppPackage.version)
    }),
  ],
  externals: [
    /^[@a-z][a-z\/\.\-0-9]*$/i, // native modules will be excluded, e.g require('react/server')
    /^.+assets\.json$/i, // these assets produced by assets-webpack-plugin
    // /^.cl$/i, // load node-opencl kernels at runtime from www
  ],
  node: {
    // fs: 'empty',
    console: true,
    __filename: true,
    __dirname: true,
  },
}


/**
 * Cordova
 */

const cordovaConfig = {
  devtool: 'source-map',
  entry: {
    client: ['./src/viz-client/index.js'],
  },
  output: {
    path: './www',
    publicPath: '/',
    filename: 'viz-client-cordova.js',
  },
  module: {
    loaders: [...commonLoadersWithPresets(['es2015', 'stage-0', 'react'])],
    noParse: [],
  },
  resolve: {
    alias: {},
  },
  postcss,
  plugins: [...commonPlugins,
    new webpack.DefinePlugin({
      __CLIENT__: true,
      __SERVER__: false,
      __CORDOVA__: true,
    }),
  ],
}

// copy static assets
const shelljs = require('shelljs')
shelljs.mkdir('-p', clientConfig.output.path)
shelljs.cp('-rf', './src/viz-client/static/*', clientConfig.output.path)
shelljs.mkdir('-p', serverConfig.output.path)
shelljs.cp('-rf', './src/viz-server/static/*', serverConfig.output.path)
shelljs.cp('-rf', './src/viz-worker/static/*', serverConfig.output.path)

const argv = process.argv[2]
if (argv === 'all' || argv === 'cordovaOnly') {
  shelljs.mkdir('-p', cordovaConfig.output.path)
  shelljs.cp('-rf', './src/viz-client/static/*', cordovaConfig.output.path)
  shelljs.mkdir('-p', serverConfig.output.path)
  shelljs.cp('-rf', './src/viz-server/static/*', cordovaConfig.output.path)
  shelljs.cp('-rf', './src/viz-worker/static/*', serverConfig.output.path)
}

module.exports = { clientConfig, serverConfig, cordovaConfig }

function commonLoadersWithPresets(presets) {
  return [{
      test: /\.jsx?$/,
      exclude: /(node_modules)/,
      loader: 'babel',
      query: {
        presets: presets.map((preset) => require.resolve(`babel-preset-${preset}`)),
        plugins: [require.resolve('babel-snabbdom-jsx'),
                  require.resolve('babel-plugin-transform-runtime')],
        cacheDirectory: true, // cache into OS temp folder by default
      }
    }, {
      test: /\.json$/,
      loader: 'json',
    }, {
      test: /\.glsl$/,
      loader: 'webpack-glsl'
    }, {
      test: /\.proto$/,
      loader: 'proto-loader'
    }, {
      test: /\.pegjs$/,
      loader: 'pegjs-loader?cache=true&optimize=size'
    }, {
      test: /\.(hbs|handlebars)$/, loader: 'handlebars-loader'
    }, {
      // match everything except [
      //   hb, js, jsx, json, css, scss, less,
      //   html, glsl, pegjs, proto, handlebars
      // ] You can add more.
      test: /\.(?!(hb|jsx?|json|s?css|less|html?|glsl|pegjs|proto|handlebars)$)([^.]+$)/,
      loader: 'url?limit=10000&name=[name]_[hash:6].[ext]',
    }, {
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
  }]
}
