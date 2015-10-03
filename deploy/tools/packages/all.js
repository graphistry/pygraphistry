//Outputs used dependencies
var fs = require('fs');

//TODO lookup package.jsons dynamically
var names = [
    'backgrid.json', 'common.json',
    'deploy.json', 'etl.json', 'marv.json', 'oneshot.json', 'package.json', 'streamgl.json',
    'central.json', 'config.json', 'dijkstra.json', 'graph.json', 'needle.json', 'opencl.json',
    'pgiz.json', 'vizserver.json'];

var sum = {};
names.forEach(function (filename) {
    var package = JSON.parse(fs.readFileSync(filename).toString());
    for (var topLevel in package) {
        sum[topLevel] = sum[topLevel] || {};
        for (var field in package[topLevel]) {
            sum[topLevel][field] = sum[topLevel][field] || package[topLevel][field];
        }
    }
});

console.log('combined package.json');
console.log(sum);

console.log('dependencies');
console.log(Object.keys(sum.dependencies).join(', '));

console.log('dev dependencies');
console.log(Object.keys(sum.devDependencies).join(', '));