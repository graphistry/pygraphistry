'use strict' //esline-disable-line

const clientConfig = require('./webpack.config.js').clientConfig
const serverConfig = require('./webpack.config.js').serverConfig

// override

/* client */
clientConfig.entry.client = ['./src/viz-client/index.test.js']
// change the client to node env, using jsdom
clientConfig.output = {
  path: './test',
  filename: 'viz-client.test.js',
}

clientConfig.node = { fs: 'empty' }

// remove the assetplugin for client
clientConfig.plugins = clientConfig.plugins.filter(p => !(p instanceof require('assets-webpack-plugin')))


/* server */
serverConfig.entry.server = ['./src/viz-server/index.test.js']
serverConfig.output = {
  path: './test',
  filename: 'viz-server.test.js',
  libraryTarget: 'commonjs2',
}

module.exports = { clientConfig, serverConfig }
