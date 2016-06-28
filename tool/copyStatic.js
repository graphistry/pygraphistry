const dir = './www'
require('shelljs').mkdir('-p', dir)
require('shelljs').cp('-rf', './src/viz-client/static/*', dir)
require('shelljs').cp('-rf', './src/viz-server/static/*', dir)
