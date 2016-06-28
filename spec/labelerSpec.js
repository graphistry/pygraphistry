'use strict';
var _ = require('underscore');
var labeler = require('../dist/labeler.js');
var Dataframe = require('../dist/Dataframe.js');


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

        dataframe.loadAttributesForType(attrs, 'point', 1);

        var expectedResult = [
            {
                "title": 0,
                "columns": [
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

        var labels = labeler.getLabels(dataframe, [0], 1);
        expect(labels).toEqual(expectedResult);
    });

});
