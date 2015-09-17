'use strict';

var PEGUtil = require('pegjs-util');
var asty    = require('asty');
var parser  = require('../src/graphVizApp/expression.js');

function parse (inputString) {
    return parser.parse(inputString);
}

function parseWithUtil (inputString) {
    return PEGUtil.parse(parser, inputString, {
        startRule: 'start',
        makeAST: function (line, column, offset, args) {
            return asty.create.apply(asty, args).pos(line, column, offset);
        }
    });
}

describe ('Numerical expressions', function () {
    it('should parse numerals', function () {
        expect(parse('3')).toEqual({type: 'Literal', value: 3});
    });
    it('should parse large numerals', function () {
        expect(parse('345679801')).toEqual({type: 'Literal', value: 345679801});
    });
    it('should parse sums', function () {
        var sum = parse('3 + 4');
        expect(sum.type).toEqual('BinaryExpression');
        expect(sum.operator).toEqual('+');
        expect(sum.left.value).toEqual(3);
        expect(sum.right.value).toEqual(4);
    });
});

describe ('literal lists', function () {
    it('should parse an empty list', function () {
        expect(parse('()')).toEqual({type: 'ListExpression', elements: []});
    });
    it('should parse single-element list', function () {
        expect(parse('(3)')).toEqual({type: 'ListExpression', elements: [{type: 'Literal', value: 3}]});
    });
    it('should parse double-element list', function () {
        expect(parse('(3, 4)')).toEqual({type: 'ListExpression', elements: [{type: 'Literal', value: 3}, {type: 'Literal', value: 4}]});
    });
    it('should parse multi-element list', function () {
        expect(parse('(3, 4, 5)')).toEqual({type: 'ListExpression', elements: [{type: 'Literal', value: 3}, {type: 'Literal', value: 4}, {type: 'Literal', value: 5}]});
    });
});

describe ('IN expressions', function () {
    it('should parse A IN B', function () {
        expect(parse('A IN B').operator).toEqual('IN');
    });
    it('should parse A in list', function () {
        var clause = parse('A IN (1, 2, 3)');
        expect(clause.operator).toBe('IN');
    });
});

describe ('Range queries', function () {
    xit('should parse A BETWEEN 2 and 5', function () {
        expect(parse('A BETWEEN 2 AND 5')).toEqual({});
    });
});

describe ('LIMIT expressions', function () {
    it('should parse LIMIT N', function () {
        expect(parse('LIMIT 4')).toEqual({type: 'Limit', value: {type: 'Literal', value: 4}});
    });
    it('should not parse LIMIT N + 1', function () {
        //expect(parse('LIMIT 4 + 3')).toThrow();
    });
});
