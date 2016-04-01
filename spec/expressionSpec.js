'use strict';

var _       = require('underscore');
var PEGUtil = require('pegjs-util');
//var asty    = require('asty');
var parser = require('../src/graphVizApp/expression.pegjs');

function parseRaw (inputString) {
    return parser.parse(inputString);
}

function parse (inputString, trace) {
    var tracer;
    if (trace) {
        tracer = {
            trace: function (info) {
                console.debug(info.type, info.rule, info.location);
            }
        };
    }
    let result = PEGUtil.parse(parser, inputString, {
        startRule: 'start',
        tracer: tracer/*,
        makeAST: function (line, column, offset, args) {
            return asty.create.apply(asty, args).pos(line, column, offset);
        }*/
    });
    if (result.error !== null) {
        throw Error(PEGUtil.errorMessage(result.error));
    }
    return result.ast;
}

describe ('Reserved literals', () => {
    it('should parse special numeric literals as string names', () => {
        expect(parse('NaN')).toEqual({type: 'Literal', dataType: 'number', value: 'NaN'});
        expect(parse('Infinity')).toEqual({type: 'Literal', dataType: 'number', value: 'Infinity'});
        expect(parse('TrUe')).toEqual({type: 'Literal', dataType: 'boolean', value: true});
        expect(parse('false')).toEqual({type: 'Literal', dataType: 'boolean', value: false});
        expect(parse('null')).toEqual({type: 'Literal', dataType: 'null', value: null});
    });
    it('should parse identifiers that start with common keywords as prefixes', () => {
        expect(parse('Indigo')).toEqual({type: 'Identifier', name: 'Indigo'});
        expect(parse('Ishtar')).toEqual({type: 'Identifier', name: 'Ishtar'});
    });
});

describe ('Numerical expressions', () => {
    it('should parse numerals', () => {
        expect(parse('3')).toEqual({type: 'Literal', dataType: 'integer', value: 3});
    });
    it('should parse large numerals', () => {
        expect(parse('345679801')).toEqual({type: 'Literal', dataType: 'integer', value: 345679801});
    });
    it('should parse sums', () => {
        expect(parse('3 + 4')).toEqual({
            type: 'BinaryExpression',
            operator: '+',
            left: {type: 'Literal', dataType: 'integer', value: 3},
            right: {type: 'Literal', dataType: 'integer', value: 4}
        });
    });
    it('should parse products', () => {
        expect(parse('3 * 4')).toEqual({
            type: 'BinaryExpression',
            operator: '*',
            left: {type: 'Literal', dataType: 'integer', value: 3},
            right: {type: 'Literal', dataType: 'integer', value: 4}
        });
    });
    it('should parse divisions', () => {
        expect(parse('3 / 4')).toEqual({
            type: 'BinaryExpression',
            operator: '/',
            left: {type: 'Literal', dataType: 'integer', value: 3},
            right: {type: 'Literal', dataType: 'integer', value: 4}
        });
    });
    it('should parse modulus', () => {
        expect(parse('3 % 4')).toEqual({
            type: 'BinaryExpression',
            operator: '%',
            left: {type: 'Literal', dataType: 'integer', value: 3},
            right: {type: 'Literal', dataType: 'integer', value: 4}
        });
    });
    it('should parse exponents', () => {
        expect(parse('3 ** 4')).toEqual({
            type: 'BinaryExpression',
            operator: '**',
            left: {type: 'Literal', dataType: 'integer', value: 3},
            right: {type: 'Literal', dataType: 'integer', value: 4}
        });
    });
});

describe ('comparison operators', () => {
    it('should parse ==', () => {
        expect(parse('a == 3').operator).toBe('==');
    });
    it('should parse <>', () => {
        expect(parse('a <> 3').operator).toBe('<>');
    });
});

describe ('literal lists', () => {
    it('should parse an empty list', () => {
        expect(parse('()')).toEqual({type: 'ListExpression', elements: []});
    });
    it('should parse single-element list', () => {
        expect(parse('(3)')).toEqual(
            {type: 'Literal', dataType: 'integer', value: 3});
        expect(parse('(3,)')).toEqual({type: 'ListExpression', elements: [
            {type: 'Literal', dataType: 'integer', value: 3}]});
    });
    it('should parse double-element list', () => {
        expect(parse('(3, 4)')).toEqual({type: 'ListExpression', elements: [
            {type: 'Literal', dataType: 'integer', value: 3},
            {type: 'Literal', dataType: 'integer', value: 4}]});
    });
    it('should parse multi-element list', () => {
        expect(parse('(3, 4, 5)')).toEqual({type: 'ListExpression', elements: [
            {type: 'Literal', dataType: 'integer', value: 3},
            {type: 'Literal', dataType: 'integer', value: 4},
            {type: 'Literal', dataType: 'integer', value: 5}]});
    });
    it('should parse with complex elements', () => {
        expect(parse('(3 + 4, 5, foo(4))')).toEqual({
            type: 'ListExpression',
            elements: [
                {
                    type: 'BinaryExpression',
                    operator: '+',
                    left: {type: 'Literal', dataType: 'integer', value: 3},
                    right: {type: 'Literal', dataType: 'integer', value: 4}
                },
                {type: 'Literal', dataType: 'integer', value: 5},
                {
                    type: 'FunctionCall',
                    callee: {type: 'FunctionIdentifier', name: 'foo'},
                    arguments: [{type: 'Literal', dataType: 'integer', value: 4}]
                }
            ]
        });
    });
});

describe ('IN expressions', () => {
    it('should parse A IN B', () => {
        expect(parse('A IN B').operator).toEqual('IN');
    });
    it('should parse A NOT IN B', () => {
        let notInParse = parse('A NOT IN B');
        expect(notInParse.operator).toEqual('NOT');
        expect(notInParse.value).toEqual(parse('A IN B'));
    });
    it('should parse A in list', () => {
        let clause = parse('A IN (1, 2, 3)');
        expect(clause.operator).toBe('IN');
        expect(_.pluck(clause.right.elements, 'value')).toEqual([1, 2, 3]);
    });
    it('should parse lower precedence than conjunctions', () => {
        let clause = parse('A IN B OR C IN D'),
            expected = parse('(A IN B) OR (C IN D)');
        expect(clause.operator).toBe(expected.operator);
        expect(clause).toEqual(expected);
    });
});

describe('LIKE expressions', () => {
    it('should handle basic LIKE parsing', () => {
        expect(parse('A LIKE "%abc%"')).toEqual({
            type: 'LikePredicate',
            operator: 'LIKE',
            left: {type: 'Identifier', name: 'A'},
            right: {type: 'Literal', dataType: 'string', value: '%abc%'}
        });
    });
    it('should handle NOT LIKE', () => {
        expect(parse('A NOT LIKE "%abc%"')).toEqual({
            type: 'NotExpression',
            operator: 'NOT',
            value: {
                type: 'LikePredicate',
                operator: 'LIKE',
                left: {type: 'Identifier', name: 'A'},
                right: {type: 'Literal', dataType: 'string', value: '%abc%'}
            }
        });
    });
    it ('should handle ESCAPE option', () => {
        expect(parse('A ILIKE "%abc%" ESCAPE "%"')).toEqual({
            type: 'LikePredicate',
            operator: 'ILIKE',
            left: {type: 'Identifier', name: 'A'},
            right: {type: 'Literal', dataType: 'string', value: '%abc%'},
            escapeChar: {type: 'Literal', dataType: 'string', value: '%'}
        });
    });
});

describe('REGEXP/SIMILAR TO expressions', () => {
    it('should parse REGEXP', () => {
        expect(parse('A REGEXP "a.*b"')).toEqual({
            type: 'RegexPredicate',
            operator: 'REGEXP',
            left: {type: 'Identifier', name: 'A'},
            right: {type: 'Literal', dataType: 'string', value: 'a.*b'}
        });
        expect(parse('A NOT REGEXP "a.*b"')).toEqual({
            type: 'NotExpression',
            operator: 'NOT',
            value: {
                type: 'RegexPredicate',
                operator: 'REGEXP',
                left: {type: 'Identifier', name: 'A'},
                right: {type: 'Literal', dataType: 'string', value: 'a.*b'}
            }
        });
    });
    it('should parse SIMILAR TO', () => {
        expect(parse('A SIMILAR TO "a.*b"')).toEqual({
            type: 'RegexPredicate',
            operator: 'SIMILAR TO',
            left: {type: 'Identifier', name: 'A'},
            right: {type: 'Literal', dataType: 'string', value: 'a.*b'}
        });
    });
});

describe ('precedence', () => {
    it('should bind * closer than +', function() {
        let clause = parse('3 + 4 * 5');
        expect(clause.operator).toBe('+');
        expect(clause.right.operator).toBe('*');
        let alt = parse('3 + (4 * 5)');
        expect(alt).toEqual(clause);
    });
    it('should use parentheses to override precedence', () => {
        let clause = parse('(3 + 4) * 5');
        expect(clause.operator).toBe('*');
        expect(clause.left.operator).toBe('+');
    });
    it('should bind comparisons closer than conjunctions', () => {
        let clause = parse('a < 4 and b > 5');
        expect(clause.operator).toBe('and');
        expect(clause.left.operator).toBe('<');
        expect(clause.right.operator).toBe('>');

        clause = parse('a <= 4 or b >= 5');
        expect(clause.operator).toBe('or');
        expect(clause.left.operator).toBe('<=');
        expect(clause.right.operator).toBe('>=');
    });
});

describe ('identifiers', () => {
    it('parses alphanumeric', () => {
        expect(parse('x')).toEqual({type: 'Identifier', name: 'x'});
        expect(parse('x_y')).toEqual({type: 'Identifier', name: 'x_y'});
        expect(parse('x_2')).toEqual({type: 'Identifier', name: 'x_2'});
    });
    it('parses colon-separated', () => {
        expect(parse('x:y')).toEqual({type: 'Identifier', name: 'x:y'});
    });
    it('parses square-bracket-quoted', () => {
        expect(parse('[x]')).toEqual(parse('x'));
        expect(parse('[x:y]')).toEqual(parse('x:y'));
        expect(parse('[x:y $]')).toEqual({
            type: 'Identifier',
            name: 'x:y $'
        });
    });
    it('parses identifiers that look like keywords', () => {
        expect(parse('endswith')).toEqual({
            type: 'Identifier', name: 'endswith'
        });
    });
    xit('parses table-scoped', () => {
        expect(parse('x.y')).toEqual({type: 'Identifier', name: 'x.y'});
    });
    it('parses multiple identifiers', () => {
        expect(parse('x - y > 0')).toEqual({
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
    });
});

describe ('NOT expressions', () => {
    it('parses nested', () => {
        let inner = parse('a');
        let one = parse('not a');
        expect(one).toEqual({
            type: 'NotExpression', operator: 'not', value: inner
        });
        let two = parse('NOT not a');
        expect(two).toEqual({
            type: 'NotExpression', operator: 'NOT', value: one
        });
    });
    it('associates more closely than binary logic', () => {
        let one = parse('not a and b');
        expect(one.operator).toBe('and');
        expect(one.left.operator).toBe('not');
        let two = parse('a or not b');
        expect(two.operator).toBe('or');
        expect(two.right.operator).toBe('not');
    });
});

describe ('IS expressions', () => {
    it('should parse special NULL tests', () => {
        let clause = parse('x ISNULL');
        expect(clause.type).toBe('UnaryExpression');
        expect(clause.operator).toBe('ISNULL');

        expect(parse('x NOTNULL').operator).toBe('NOTNULL');
    });
    it('should parse IS keyword comparisons', () => {
        expect(parse('x IS TRUE')).toEqual({
            type: 'BinaryPredicate',
            operator: 'IS',
            left: {type: 'Identifier', name: 'x'},
            right: {
                type: 'Literal', dataType: 'boolean', value: true
            }
        });
        expect(parse('x IS FALSE')).toEqual({
            type: 'BinaryPredicate',
            operator: 'IS',
            left: {type: 'Identifier', name: 'x'},
            right: {
                type: 'Literal', dataType: 'boolean', value: false
            }
        });
        expect(parse('x IS NULL')).toEqual({
            type: 'BinaryPredicate',
            operator: 'IS',
            left: {type: 'Identifier', name: 'x'},
            right: {
                type: 'Literal', dataType: 'null', value: null
            }
        });
    });
    it('should parse negative IS comparisons', () => {
        expect(parse('x IS NOT NULL')).toEqual({
            type: 'NotExpression',
            operator: 'NOT',
            value: {
                type: 'BinaryPredicate',
                operator: 'IS',
                left: {type: 'Identifier', name: 'x'},
                right: {type: 'Literal', dataType: 'null', value: null}
            }
        });
    });
});

describe ('member access', () => {
    it('should parse after identifiers', function() {
        expect(parse('a[4]')).toEqual({
            type: 'MemberAccess',
            object: {type: 'Identifier', name: 'a'},
            property: {type: 'Literal', dataType: 'integer', value: 4}
        });
    });
    it('should parse with whitespace', function() {
        expect(parse('a [4]')).toEqual(parse('a[4]'));
    });
    it('should nest', () => {
        expect(parse('a[b[4]]')).toEqual({
            type: 'MemberAccess',
            object: {type: 'Identifier', name: 'a'},
            property: {
                type: 'MemberAccess',
                object: {type: 'Identifier', name: 'b'},
                property: {type: 'Literal', dataType: 'integer', value: 4}
            }
        });
    });
    it('should chain', () => {
        expect(parse('a[3][4]')).toEqual({
            type: 'MemberAccess',
            object: {
                type: 'MemberAccess',
                object: {type: 'Identifier', name: 'a' },
                property: {type: 'Literal', dataType: 'integer', value: 3}
            },
            property: {type: 'Literal', dataType: 'integer', value: 4}
        });
    });
});

describe ('function calls', () => {
    it('should work with one argument', () => {
        expect(parse('substr(1)')).toEqual({
            type: 'FunctionCall',
            callee: {type: 'FunctionIdentifier', name: 'substr'},
            arguments: [{type: 'Literal', dataType: 'integer', value: 1}]
        });
    });
    it('should parse with a space before arguments', () => {
        expect(parse('substr (1)')).toEqual(parse('substr(1)'));
    });
    it('should handle empty', () => {
        expect(parse('substr()')).toEqual({
            type: 'FunctionCall',
            callee: {type: 'FunctionIdentifier', name: 'substr'},
            arguments: []
        });
    });
    it('should handle argument lists', () => {
        expect(parse('substr("abcdef", 3, 4)')).toEqual({
            type: 'FunctionCall',
            callee: {type: 'FunctionIdentifier', name: 'substr'},
            arguments: [
                {type: 'Literal', dataType: 'string', value: 'abcdef'},
                {type: 'Literal', dataType: 'integer', value: 3},
                {type: 'Literal', dataType: 'integer', value: 4}
            ]
        });
    });
    it('should nest', () => {
        expect(parse('length(substring)')).toEqual({
            type: 'FunctionCall',
            callee: {type: 'FunctionIdentifier', name: 'length'},
            arguments: [{type: 'Identifier', name: 'substring'}]
        });
        expect(parse('length(substring())')).toEqual({
            type: 'FunctionCall',
            callee: {type: 'FunctionIdentifier', name: 'length'},
            arguments: [{
                type: 'FunctionCall',
                callee: {type: 'FunctionIdentifier', name: 'substring'},
                arguments: []
            }]
        });
        expect(parse('length(substring("abcdef"))')).toEqual({
            type: 'FunctionCall',
            callee: {type: 'FunctionIdentifier', name: 'length'},
            arguments: [{
                type: 'FunctionCall',
                callee: {type: 'FunctionIdentifier', name: 'substring'},
                arguments: [{type: 'Literal', dataType: 'string', value: 'abcdef'}]
            }]
        });
    });
});

describe ('Range predicates', () => {
    it('should parse A BETWEEN 2 and 5', () => {
        let betweenAnd = parse('A BETWEEN 2 AND 5');
        expect(betweenAnd).toEqual({
            type: 'BetweenPredicate',
            value: {type: 'Identifier', name: 'A'},
            start: {type: 'Literal', dataType: 'integer', value: 2},
            stop: {type: 'Literal', dataType: 'integer', value: 5}
        });
    });
    it('should parse A NOT BETWEEN 2 and 5', () => {
        let betweenAnd = parse('A BETWEEN 2 AND 5');
        expect(parse('A NOT BETWEEN 2 AND 5')).toEqual({
            type: 'NotExpression',
            operator: 'NOT',
            value: betweenAnd
        });
    });
});

describe ('LIMIT clauses', () => {
    it('should parse LIMIT N', () => {
        expect(parse('LIMIT 4')).toEqual({type: 'Limit', value: {type: 'Literal', dataType: 'integer', value: 4}});
    });
    it('should not parse LIMIT N + 1', () => {
        expect(parse('LIMIT 4 + 3')).toEqual({
            type: 'Limit',
            value: {
                type: 'BinaryExpression',
                operator: '+',
                left: {type: 'Literal', dataType: 'integer', value: 4},
                right: {type: 'Literal', dataType: 'integer', value: 3}
            }
        });
    });
});

describe('IN/MEMBEROF expressions', () => {
    it('should parse IN', () => {
        expect(parse('IN foo')).toEqual({
            type: 'MemberOfExpression',
            operator: 'IN',
            value: {type: 'Identifier', name: 'foo'}
        });
    });
    it('should parse IN', () => {
        expect(parse('MEMBEROF foo')).toEqual({
            type: 'MemberOfExpression',
            operator: 'MEMBEROF',
            value: {type: 'Identifier', name: 'foo'}
        });
    });
});

describe ('IF/THEN/ELSE expressions', () => {
    it('should parse IF/THEN/END', () => {
        expect(parse('IF true THEN false END', true)).toEqual({
            type: 'ConditionalExpression',
            cases: [
                {
                    type: 'CaseBranch',
                    condition: {type: 'Literal', dataType: 'boolean', value: true},
                    result: {type: 'Literal', dataType: 'boolean', value: false}
                }
            ], elseClause: undefined
        });
    });
    it('should parse IF/THEN/ELSE/END', () => {
        expect(parse('IF true THEN false ELSE true END', true)).toEqual({
            type: 'ConditionalExpression',
            cases: [
                {
                    type: 'CaseBranch',
                    condition: {type: 'Literal', dataType: 'boolean', value: true},
                    result: {type: 'Literal', dataType: 'boolean', value: false}
                }
            ], elseClause: {type: 'Literal', dataType: 'boolean', value: true}
        });
    });
    it('should parse chained IF clauses', () => {
        expect(parse('IF true THEN 1 ELSE IF FALSE THEN 2 END', true)).toEqual({
            type: 'ConditionalExpression',
            cases: [
                {
                    type: 'CaseBranch',
                    condition: {type: 'Literal', dataType: 'boolean', value: true},
                    result: {type: 'Literal', dataType: 'integer', value: 1}
                },
                {
                    type: 'CaseBranch',
                    condition: {type: 'Literal', dataType: 'boolean', value: false},
                    result: {type: 'Literal', dataType: 'integer', value: 2}
                }
            ], elseClause: undefined
        });
        expect(parse('IF true THEN 1 ELSE IF FALSE THEN 2 ELSE 3 END', true)).toEqual({
            type: 'ConditionalExpression',
            cases: [
                {
                    type: 'CaseBranch',
                    condition: {type: 'Literal', dataType: 'boolean', value: true},
                    result: {type: 'Literal', dataType: 'integer', value: 1}
                },
                {
                    type: 'CaseBranch',
                    condition: {type: 'Literal', dataType: 'boolean', value: false},
                    result: {type: 'Literal', dataType: 'integer', value: 2}
                }
            ], elseClause: {type: 'Literal', dataType: 'integer', value: 3}
        });
    });
});

describe ('CASE expressions', () => {
    it('should parse with one rule', () => {
        expect(parse('CASE WHEN true THEN false ELSE true END')).toEqual({
            type: 'CaseExpression',
            value: undefined,
            cases: [{
                type: 'CaseBranch',
                condition: {type: 'Literal', dataType: 'boolean', value: true},
                result: {type: 'Literal', dataType: 'boolean', value: false}}],
            elseClause: {type: 'Literal', dataType: 'boolean', value: true}
        });
    });
    it('should parse with one rule on a variable', () => {
        expect(parse('CASE x WHEN true THEN 1 END')).toEqual({
            type: 'CaseExpression',
            value: {type: 'Identifier', name: 'x'},
            cases: [{
                type: 'CaseBranch',
                condition: {type: 'Literal', dataType: 'boolean', value: true},
                result: {type: 'Literal', dataType: 'integer', value: 1}
            }],
            elseClause: undefined
        });
    });
    it('should parse with more rules', () => {
        expect(parse('CASE WHEN true THEN 1 WHEN false THEN 2 ELSE 3 END')).toEqual({
            type: 'CaseExpression',
            value: undefined,
            cases: [
                {
                    type: 'CaseBranch',
                    condition: {type: 'Literal', dataType: 'boolean', value: true},
                    result: {type: 'Literal', dataType: 'integer', value: 1}
                },
                {
                    type: 'CaseBranch',
                    condition: {type: 'Literal', dataType: 'boolean', value: false},
                    result: {type: 'Literal', dataType: 'integer', value: 2}
                }
            ],
            elseClause: {type: 'Literal', dataType: 'integer', value: 3}
        });
    });
});
