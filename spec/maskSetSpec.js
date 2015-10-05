'use strict';

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
});

describe('complement', function () {
    var universe;
    beforeEach(function () {
        universe = jasmine.createSpy('universe', 'getPoints');
    });
});
