'use strict';
var _ = require('underscore');
var labeler = require('../js/labeler.js');
var Dataframe = require('../js/Dataframe.js');


describe('Labeler', function () {

    it('should return default labels if none are pre-specified', function () {
        var dataframe = new Dataframe();

        var attrs = {
            numAttr: {
                name: 'numAttr', target: 0, type: 'number', values: [5]
            },
            strAttr: {
                name: 'strAttr', target: 0, type: 'string', values: ['bar']
            }
        };

        dataframe.load(attrs, 'point', 1);

        var expectedResult = [
            {
                "title": 0,
                "columns": [
                    {
                        "value": 0,
                        "key": "_index"
                    },
                    {
                        "value": 5,
                        "key": "numAttr",
                        "dataType": "number"
                    },
                    {
                        "value": "bar",
                        "key": "strAttr",
                        "dataType": "string"
                    }
                ]
            }
        ];

        // Mock graph object
        var graph = {
            simulator: {
                dataframe: dataframe
            },
            dataframe: dataframe
        };

        var labels = labeler.getLabels(graph, [0], 1);
        expect(labels).toEqual(expectedResult);
    });

});
