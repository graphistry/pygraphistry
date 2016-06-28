'use strict' //eslint-disable-line

/**
 * Dependencies
 */

const path = require('path')

// get the last argument, see the dev.js
const argv = process.argv[2]

const fs = require('fs')

const webpack = require('webpack')
const ExtractTextPlugin = require('extract-text-webpack-plugin')

const clientConfig = require('./webpack.config.js').clientConfig
const serverConfig = require('./webpack.config.js').serverConfig

const commonLoaders = [
  { test: /\.css$/, loader: ExtractTextPlugin.extract('css?module&minimize&localIdentName=[local]_[hash:6]!postcss') },
  { test: /\.less$/, loader: ExtractTextPlugin.extract('css?module&minimize&localIdentName=[local]_[hash:6]!postcss!less') },
]

const commonPlugins = [
  new webpack.DefinePlugin({
    __DEV__: false,
    'process.env.NODE_ENV': '"production"',
  }),
  new ExtractTextPlugin('styles_[contenthash:6].css', {
    allChunks: true,
  }),
  new webpack.optimize.DedupePlugin(),
  new webpack.optimize.AggressiveMergingPlugin(),
  new webpack.optimize.UglifyJsPlugin({
    compress: {
      warnings: false,
    },
    sourceMap: false,
    comments: false,
    'screw-ie8': true,
  }),
]


/**
 * Client
 */

clientConfig.output.filename = 'viz-client.js'
clientConfig.module.loaders.push(...commonLoaders)
clientConfig.plugins.push(...commonPlugins)
;(argv !== 'cordovaOnly') && webpack(clientConfig).run((err, stats) => { // eslint-disable-line no-unused-expressions
  console.log('Client Bundles \n', stats.toString({
    colors: true,
  }), '\n')
    // cssnano, temparory work around
  try {
    const fileName = require(path.resolve('www/webpack-assets.json')).client.css
    const filePath = path.resolve('www', fileName.replace(/^\/+/, ''))
    const css = fs.readFileSync(filePath)
    require('cssnano').process(css, { discardComments: { removeAll: true } }).then((result) => {
      require('fs').writeFileSync(filePath, result.css)
    })
  } catch (e) { /* do nothing */ }
})

/**
 * Server
 */

serverConfig.module.loaders.push(...commonLoaders)
serverConfig.plugins.push(...commonPlugins)
;(argv !== 'cordovaOnly') && webpack(serverConfig).run((err, stats) => { // eslint-disable-line no-unused-expressions
  console.log('Server Bundle \n', stats.toString({
    colors: true,
  }), '\n')
  require('child_process').exec('rm www/styles_??????.css', () => {})
  // then delele the styles.css in the server folder
  // try {
  // const styleFile = _root+'/www/styles.css'
  //  fs.statSync(styleFile) && fs.unlinkSync(styleFile)
  // } catch(e) {/*do nothing*/}
  // file loader may also result in duplicated files from shared React components
})

/**
 * Cordova
 */

const cordovaConfig = require('./webpack.config.js').cordovaConfig

cordovaConfig.module.loaders.push(...commonLoaders)
cordovaConfig.plugins.push(...commonPlugins)

// remove the ExtractTextPlugin
cordovaConfig.plugins = cordovaConfig.plugins.filter(p => !(p instanceof ExtractTextPlugin))
// re-add the ExtractTextPlugin with new option
cordovaConfig.plugins.push(new ExtractTextPlugin('styles.css', { allChunks: true }))

;(argv === 'all' || argv === 'cordovaOnly') && webpack(cordovaConfig).run((err, stats) => {  // eslint-disable-line no-unused-expressions
  console.log('Cordova Bundles \n', stats.toString({
    colors: true,
  }), '\n')
    // cssnano, temparory work around
  try {
    const filePath = path.resolve(cordovaConfig.output.path, 'styles.css')
    const css = fs.readFileSync(filePath)
    require('cssnano').process(css, { discardComments: { removeAll: true } }).then((result) => {
      require('fs').writeFileSync(filePath, result.css)
    })
  } catch (e) { /* do nothing */ }
})
