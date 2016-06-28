'use strict' // eslint-disable-line

/**
 * Dependencies
 */

const path = require('path')
// const _root = path.resolve()
const webpack = require('webpack')

const clientConfig = require('./webpack.config.test.js').clientConfig
const serverConfig = require('./webpack.config.test.js').serverConfig

const ExtractTextPlugin = require('extract-text-webpack-plugin')

const commonLoaders = [
  { test: /\.css$/, loader: ExtractTextPlugin.extract('css?module&localIdentName=[local]_[hash:6]!postcss') },
]

const commonPlugins = [
  new webpack.DefinePlugin({
    __DEV__: true,
    __TEST__: true,
    'process.env.NODE_ENV': '"development"',
  }),
  new ExtractTextPlugin('styles.css', { allChunks: true }), // has to use this for universal server client rendering
]

/**
 * Client
 */

// clientConfig.module.loaders.push(...)
// clientConfig.plugins.push(...)

compileAndTest(clientConfig, 'CLIENT')

/**
 * Server
 */

// serverConfig.module.loaders.push(...)
// serverConfig.plugins.push(...)

compileAndTest(serverConfig, 'SERVER')


/**
 * @arg {Object} config
 * @arg {string} arch - 'CLIENT' | 'SERVER'
 */

function compileAndTest(config, arch) {
  config.devtool = 'cheap-module-eval-source-map'
  config.module.loaders.push(...commonLoaders)
  config.plugins.push(...commonPlugins)

  let child
  webpack(config).watch({}, (err, stats) => {
    if (stats.hasErrors()) return console.log(arch + '\n', stats.toString({ colors: true }))
    if (child) child.kill()
    child = require('child_process').fork(path.join(config.output.path, config.output.filename))
  })
}
