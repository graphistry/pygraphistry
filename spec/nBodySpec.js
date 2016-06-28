'use strict';

var _ = require('underscore');
var NBody = require('../dist/NBody.js');

describe('nBody', function () {
    var minimalSim, minimalSocket;

    beforeEach(function () {
        minimalSim = {
            controls: {},
            dataframe: {}
        };

        minimalSocket = {
            emit: _.identity
        };
    });


    it('should initialize', function (done) {
        NBody.create({}, minimalSim, {}, 'device', 'vendor', {}, minimalSocket)
            .then(function (graph) {
                done();
            });
    });

    it('should expand midedge colors', function (done) {
        var colors = [1,2,3];
        var expanded = new Uint32Array([1,1,2,2,3,3]);
        minimalSim.setMidEdgeColors = jasmine.createSpy('setMidEdgeColors');
        minimalSim.dataframe.getNumElements = jasmine.createSpy('getNumElements').andReturn(3);

        NBody.create({}, minimalSim, {}, 'device', 'vendor', {}, minimalSocket)
            .then(function (graph) {
                graph.setMidEdgeColors(colors);
                expect(minimalSim.setMidEdgeColors.mostRecentCall.args[0])
                    .toEqual(expanded);
                done();
            });
    });

});
