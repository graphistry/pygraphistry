'use strict';
var _ = require('underscore');
var labeler = require('../js/labeler.js');


describe('Labeler', function () {

    it('should return default labels if none are pre-specified', function () {

        var graph = {
            simulator: {},
            dataframe: {
                getLabels: _.identity,
                getDataType: _.identity,
                getRows: _.identity,
                doesColumnRepresentColorPaletteMap: _.identity
            }
        };
        graph.simulator.dataframe = graph.dataframe;

        var rows = [
            {weight: 5, _title: 9, Source: 'Valjean', _index: 9}
        ];

        var expectedResult = [
          {
            "title": 9,
            "columns": [
              {
                "value": "Valjean",
                "key": "Source",
                "dataType": "string"
              },
              {
                "value": 9,
                "key": "_index"
              },
              {
                "value": 5,
                "key": "weight",
                "dataType": "number"
              }
            ]
          }
        ];

        spyOn(graph.dataframe, 'getLabels').andReturn(undefined);
        spyOn(graph.dataframe, 'getRows').andReturn(rows);
        spyOn(graph.dataframe, 'doesColumnRepresentColorPaletteMap').andReturn(false);
        spyOn(graph.dataframe, 'getDataType').andCallFake(function (name) {
            var type;
            switch (name) {
                case 'weight':
                case '_title':
                    type = 'number';
                    break;
                case 'Source':
                    type = 'string';
                    break;
            }
            return type;
        });

        var labels = labeler.getLabels(graph, [1], 1);
        expect(labels).toEqual(expectedResult);
    });

});
