'use strict';

var _       = require('underscore');
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
        expect(parse('(3)')).toEqual({type: 'Literal', value: 3});
        expect(parse('(3,)')).toEqual({type: 'ListExpression', elements: [{type: 'Literal', value: 3}]});
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
        expect(_.pluck(clause.right.elements, 'value')).toEqual([1, 2, 3]);
    });
});

describe ('precedence', function () {
    it('should bind * closer than +', function() {
        var clause = parse('3 + 4 * 5');
        expect(clause.operator).toBe('+');
        expect(clause.right.operator).toBe('*');
        var alt = parse('3 + (4 * 5)');
        expect(alt).toEqual(clause);
    });
    it('should use parentheses to override precedence', function () {
        var clause = parse('(3 + 4) * 5');
        expect(clause.operator).toBe('*');
        expect(clause.left.operator).toBe('+');
    });
    it('should bind comparisons closer than conjunctions', function () {
        var clause = parse('a < 4 and b > 5');
        expect(clause.operator).toBe('and');
        expect(clause.left.operator).toBe('<');
        expect(clause.right.operator).toBe('>');

        clause = parse('a <= 4 or b >= 5');
        expect(clause.operator).toBe('or');
        expect(clause.left.operator).toBe('<=');
        expect(clause.right.operator).toBe('>=');
    });
});

describe ('identifiers', function () {
    it('parses alphanumeric', function () {
        expect(parse('x')).toEqual({type: 'Identifier', name: 'x'});
        expect(parse('x_y')).toEqual({type: 'Identifier', name: 'x_y'});
        expect(parse('x_2')).toEqual({type: 'Identifier', name: 'x_2'});
    });
    it('parses colon-separated', function () {
        expect(parse('x:y')).toEqual({type: 'Identifier', name: 'x:y'});
    });
    xit('parses table-scoped', function () {
        expect(parse('x.y')).toEqual({type: 'Identifier', name: 'x.y'});
    });
});

describe ('NOT expressions', function () {
    it('parses nested', function () {
        var inner = parse('a');
        var one = parse('not a');
        expect(one).toEqual({
            type: 'UnaryExpression', operator: 'not', fixity: 'prefix', argument: inner
        });
        var two = parse('NOT not a');
        expect(two).toEqual({
            type: 'UnaryExpression', operator: 'NOT', fixity: 'prefix', argument: one
        });
    });
    it('associates more closely than binary logic', function () {
        var one = parse('not a and b');
        expect(one.operator).toBe('and');
        expect(one.left.operator).toBe('not');
        var two = parse('a or not b');
        expect(two.operator).toBe('or');
        expect(two.right.operator).toBe('not');
    });
});

describe ('IS expressions', function () {
    it('should parse special NULL tests', function () {
        var clause = parse('x ISNULL');
        expect(clause.type).toBe('UnaryExpression');
        expect(clause.operator).toBe('ISNULL');

        expect(parse('x NOTNULL').operator).toBe('NOTNULL');
    });
    it('should parse IS keyword comparisons', function () {
        expect(parse('x IS TRUE')).toEqual({
            type: 'LogicalExpression',
            operator: 'IS',
            left: {type: 'Identifier', name: 'x'},
            right: {
                type: 'Literal', value: true
            }
        });
        expect(parse('x IS FALSE')).toEqual({
            type: 'LogicalExpression',
            operator: 'IS',
            left: {type: 'Identifier', name: 'x'},
            right: {
                type: 'Literal', value: false
            }
        });
        expect(parse('x IS NULL')).toEqual({
            type: 'LogicalExpression',
            operator: 'IS',
            left: {type: 'Identifier', name: 'x'},
            right: {
                type: 'Literal', value: null
            }
        });
    });
    it('should parse negative IS comparisons', function () {
        expect(parse('x IS NOT NULL')).toEqual({
            type: 'LogicalExpression',
            operator: 'IS',
            left: {type: 'Identifier', name: 'x'},
            right: {type: 'NotExpression', operator: 'NOT', value: {type: 'Literal', value: null}}
        });
    });
});

describe ('function calls', function () {
    it('should parse', function () {
        var clause = parse('foobar()');
        expect(clause).toEqual({
            type: 'FunctionCall',
            callee: {type: 'Identifier', name: 'foobar'},
            arguments: []
        });
        expect(clause).toEqual(parse('foobar ()'));
    });
    it('should handle argument lists', function () {
        expect(parse('substr("abcdef", 3, 4)')).toEqual({
            type: 'FunctionCall',
            callee: {type: 'Identifier', name: 'substr'},
            arguments: [
                {type: 'Literal', value: 'abcdef'},
                {type: 'Literal', value: 3},
                {type: 'Literal', value: 4}
            ]
        });
    });
});

describe ('Range queries', function () {
    it('should parse A BETWEEN 2 and 5', function () {
        var betweenAnd = parse('A BETWEEN 2 AND 5');
        expect(betweenAnd).toEqual({
            type: 'BetweenAndExpression',
            value: {type: 'Identifier', name: 'A'},
            start: {type: 'Literal', value: 2},
            stop: {type: 'Literal', value: 5}
        });
    });
    it('should parse A NOT BETWEEN 2 and 5', function () {
        var betweenAnd = parse('A BETWEEN 2 AND 5');
        expect(parse('A NOT BETWEEN 2 AND 5')).toEqual({
            type: 'UnaryExpression',
            operator: 'not',
            value: betweenAnd
        });
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
