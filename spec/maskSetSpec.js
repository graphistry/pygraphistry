'use strict';

var _ = require('underscore');

var MaskSet = require('../js/MaskSet');
//var Dataframe = require('../js/Dataframe');

describe('union', function () {
    it('handles empty masks', function () {
        expect(MaskSet.unionOfTwoMasks([], [])).toEqual([]);
    });
    it('handles singletons', function () {
        expect(MaskSet.unionOfTwoMasks([3], [4])).toEqual([3, 4]);
        expect(MaskSet.unionOfTwoMasks([4], [3])).toEqual([3, 4]);
        expect(MaskSet.unionOfTwoMasks([3], [3])).toEqual([3]);
    });
    it('handles disjoint masks', function () {
        expect(MaskSet.unionOfTwoMasks([3, 7], [2, 5])).toEqual([2, 3, 5, 7]);
        expect(MaskSet.unionOfTwoMasks([], [2, 5])).toEqual([2, 5]);
    });
});

describe('intersection', function () {
    it('handles empty masks', function () {
        expect(MaskSet.intersectionOfTwoMasks([], [])).toEqual([]);
    });
    it('handles singletons', function () {
        expect(MaskSet.intersectionOfTwoMasks([3], [4])).toEqual([]);
        expect(MaskSet.intersectionOfTwoMasks([4], [3])).toEqual([]);
        expect(MaskSet.intersectionOfTwoMasks([3], [3])).toEqual([3]);
    });
    it('handles disjoint masks', function () {
        expect(MaskSet.intersectionOfTwoMasks([3, 7], [2, 5])).toEqual([]);
        expect(MaskSet.intersectionOfTwoMasks([], [2, 5])).toEqual([]);
    });
});

describe('complement', function () {
    it('handles empty masks', function () {
        expect(MaskSet.complementOfMask([], 100)).toEqual(_.range(100));
    });
    it('handles singletons', function () {
        expect(MaskSet.complementOfMask([3], 5)).toEqual([0,1,2,4]);
        expect(MaskSet.complementOfMask([0], 5)).toEqual([1,2,3,4]);
        expect(MaskSet.complementOfMask([5], 5)).toEqual([0,1,2,3,4]);
    });
    it('handles complex patterns', function () {
        expect(MaskSet.complementOfMask([0,1,2,3,4], 5)).toEqual([]);
        expect(MaskSet.complementOfMask([0,1], 5)).toEqual([2,3,4]);
        expect(MaskSet.complementOfMask([0,4], 5)).toEqual([1,2,3]);
    });
    it('ignores elements outside the universe', function () {
        expect(MaskSet.complementOfMask([6], 5)).toEqual([0,1,2,3,4]);
        expect(MaskSet.complementOfMask([5], 5)).toEqual([0,1,2,3,4]);
        //expect(MaskSet.complementOfMask([-1], 5)).toEqual([0,1,2,3,4]);
    });
});
