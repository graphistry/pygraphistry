'use strict';

var ExpressionCodeGenerator = require('../dist/expressionCodeGenerator');

describe('Regular expressions from LIKE patterns', function () {
    var codeGenerator = new ExpressionCodeGenerator();
    it('should not transform patterns without placeholders', function () {
        expect(codeGenerator.regularExpressionLiteralFromLikePattern('abc')).toBe('abc');
    });
    it('should transform % to .*', function () {
        expect(codeGenerator.regularExpressionLiteralFromLikePattern('a%b')).toBe('a.*b');
        expect(codeGenerator.regularExpressionLiteralFromLikePattern('%ab')).toBe('.*ab');
        expect(codeGenerator.regularExpressionLiteralFromLikePattern('ab%')).toBe('ab.*');
        expect(codeGenerator.regularExpressionLiteralFromLikePattern('%ab%')).toBe('.*ab.*');
    });
    it('should transform _ to .', function () {
        expect(codeGenerator.regularExpressionLiteralFromLikePattern('a_b')).toBe('a.b');
        expect(codeGenerator.regularExpressionLiteralFromLikePattern('_ab')).toBe('.ab');
        expect(codeGenerator.regularExpressionLiteralFromLikePattern('ab_')).toBe('ab.');
        expect(codeGenerator.regularExpressionLiteralFromLikePattern('__ab')).toBe('..ab');
        expect(codeGenerator.regularExpressionLiteralFromLikePattern('a__b')).toBe('a..b');
    });
    xit('should quote', function () {
        expect(codeGenerator.regularExpressionLiteralFromLikePattern('a%%b')).toBe('a%b');
        expect(codeGenerator.regularExpressionLiteralFromLikePattern('a%_b')).toBe('a_b');
    });
    it('should quote periods', function () {
        expect(codeGenerator.regularExpressionLiteralFromLikePattern('ab.c')).toBe('ab[.]c');
    });
});
