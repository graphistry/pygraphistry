'use strict';

var _ = require('underscore');

var ExpressionPlan = require('../dist/ExpressionPlan.js');
var Dataframe = require('../dist/Dataframe.js');

describe('Plans for multi-column expressions', function () {
    var dataframe;
    beforeEach(function () {
        dataframe = new Dataframe();
        spyOn(dataframe, 'getDataType').andCallFake(function (/*attributeName, type*/) {
            return 'number';
        });
        spyOn(dataframe, 'normalizeAttributeName').andCallFake(function (attributeName) {
            return {type: 'point', attribute: attributeName};
        });
    });
    it('should work', function () {
        var actual = new ExpressionPlan(dataframe, {
            type: 'BinaryPredicate',
            operator: '>',
            left: {
                type: 'BinaryExpression',
                operator: '-',
                left: {type: 'Identifier', name: 'x'},
                right: {type: 'Identifier', name: 'y'}
           },
            right: {
                type: 'Literal',
                dataType: 'integer',
                value: 0
           }
        });
        expect(actual.rootNode.arity()).toEqual(2);
        expect(actual.rootNode.returnType()).toEqual('Positions');
        var expected = {
            x: [
                {
                    ast: {type: 'Identifier', name: 'x'},
                    inputNodes: [],
                    attributeData: {x: {type: 'point', attribute: 'x'}},
                    bindings: {attributes: []},
                    executor: Function
                }],
            y: [
                {
                    ast: {type: 'Identifier', name: 'y'},
                    inputNodes: [],
                    attributeData: {y: {type: 'point', attribute: 'y'}},
                    bindings: {attributes: []},
                    executor: Function
                }]
        };
        var got = _.mapObject(actual.rootNode.identifierNodes(), function (planNode) {
            return planNode;
        });
        expect(_.keys(got)).toEqual(['x', 'y']);
        expect(got.x.length).toBe(1);
        expect(got.x[0].attributeData).toEqual(expected.x[0].attributeData);
        expect(got.y.length).toBe(1);
        expect(got.y[0].attributeData).toEqual(expected.y[0].attributeData);
    });
});
