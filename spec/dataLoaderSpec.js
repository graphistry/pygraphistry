'use strict';

var loader = require('../dist/data-loader.js');

describe('Data loader', function () {

    it('should download s3 vgraph datasets', function (done) {
        var config = {
            name: 'Miserables',
            url: 'Miserables'
        };
        var expectedResponseLength = 3237;

        var qDataset = loader.downloadDataset(config);
        qDataset.then(function (resp) {
            expect(resp.body.length).toEqual(expectedResponseLength);
            done();
        });

    });

});
