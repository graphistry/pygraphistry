'use strict';

var Dataframe = require('../dist/Dataframe.js');
var ComputedColumnManager = require('../dist/ComputedColumnManager.js');
var ComputedColumnSpec = require('../dist/ComputedColumnSpec.js');


describe('Computed Columns', function () {
    var testDF;
    var ccManager;

    beforeEach(function () {
        testDF = new Dataframe();

        // LOAD DATAFRAME
        var aVals = [0,1,2,3,4];
        var bVals = [0,2,4,6,8];
        var cVals = ['zero', 'one', 'two', 'three', 'four'];
        var dVals = [0,0,1,1,2,2,3,3,4,4];

        var aValueObj = {name: 'a', values: aVals, type: 'number'};
        var bValueObj = {name: 'b', values: bVals, type: 'number'};
        var cValueObj = {name: 'c', values: cVals, type: 'string'};
        var dValueObj = {name: 'd', values: dVals, type: 'number', numberPerGraphComponent: 2};

        var attrObjects = {
            a: aValueObj,
            b: bValueObj,
            c: cValueObj,
            d: dValueObj
        };

        testDF.loadAttributesForType(attrObjects, 'point', 5);

        // LOAD BASIC COMPUTED COLUMNS
        ccManager = testDF.computedColumnManager;

        var baseDesc = new ComputedColumnSpec({
            type: 'number',
            graphComponentType: 'point'
        });

        var eDesc = baseDesc.clone();
        eDesc.setDependencies([
            ['a', 'point'],
            ['b', 'point']
        ]);
        eDesc.setComputeSingleValue(function (a, b) {
            return a + b;
        });

        var fDesc = baseDesc.clone();
        fDesc.setDependencies([
            ['e', 'point']
        ]);
        fDesc.setComputeSingleValue(function (e) {
            return e*2;
        });

        ccManager.addComputedColumn(testDF, 'point', 'e', eDesc);
        ccManager.addComputedColumn(testDF, 'point', 'f', fDesc);

    });

    it('should compute basic values', function () {
        expect(testDF.getCell(2, 'point', 'e')).toEqual(6);
        expect(testDF.getColumnValues('e', 'point')).toEqual([0,3,6,9,12]);
        expect(testDF.getCell(2, 'point', 'e')).toEqual(6);
    });

    it('should compute values that depend on other computed columns', function () {
        expect(testDF.getCell(2, 'point', 'f')).toEqual(12);
        expect(testDF.getColumnValues('f', 'point')).toEqual([0,6,12,18,24]);
        expect(testDF.getCell(2, 'point', 'f')).toEqual(12);
    });

    it('should work when only compute all values is set', function () {
        var newDesc = ccManager.getComputedColumnSpec('point', 'f').clone();
        newDesc.setDependencies([['e', 'point']]);
        newDesc.setComputeAllValues(function (eVals, outArr, numGraphElements) {
            for (var i = 0; i < outArr.length; i++) {
                outArr[i] = eVals[i] * 2;
            }
        });
        ccManager.addComputedColumn(testDF, 'point', 'f', newDesc);

        expect(testDF.getCell(2, 'point', 'f')).toEqual(12);
        expect(testDF.getColumnValues('f', 'point')).toEqual([0,6,12,18,24]);
        expect(testDF.getCell(2, 'point', 'f')).toEqual(12);
    });

    it('should update dependent computed columns', function () {

        expect(testDF.getCell(2, 'point', 'f')).toEqual(12);
        expect(testDF.getColumnValues('f', 'point')).toEqual([0,6,12,18,24]);
        expect(testDF.getCell(2, 'point', 'f')).toEqual(12);

        var newDesc = ccManager.getComputedColumnSpec('point', 'e').clone();
        newDesc.setDependencies([['a', 'point'], ['b', 'point']]);
        newDesc.setComputeSingleValue(function (a, b) {
            return a+b+1;
        });
        ccManager.addComputedColumn(testDF, 'point', 'e', newDesc);

        expect(testDF.getCell(2, 'point', 'f')).toEqual(14);
        expect(testDF.getColumnValues('f', 'point')).toEqual([2,8,14,20,26]);
        expect(testDF.getCell(2, 'point', 'f')).toEqual(14);
    });

    it('should not let you cause a dependency cycle', function () {

        var shouldThrow = function () {

            var newDesc = ccManager.getComputedColumnSpec('point', 'e').clone();
            newDesc.setDependencies([['a', 'point'], ['f', 'point']]);
            newDesc.setComputeSingleValue(function (a, f) {
                return a+f;
            });
            ccManager.addComputedColumn(testDF, 'point', 'e', newDesc);

        }

        expect(shouldThrow).toThrow();

    });

    it('should not let you remove a computed columns dependency', function () {

        var shouldThrow = function () {
            ccManager.removeComputedColumnInternally('point', 'e');
        }

        expect(shouldThrow).toThrow();

    });

    it('should work for columns that have more than one value per graph element', function () {

        var baseDesc = new ComputedColumnSpec({
            type: 'number',
            graphComponentType: 'point',
            numberPerGraphComponent: 2,
        });

        var singleValueDesc = baseDesc.clone();
        singleValueDesc.setDependencies([['d', 'point']]);
        singleValueDesc.setComputeSingleValue(function (vals) {
            return vals.map(function (v) {
                return v*2;
            });
        });

        var multiValueDesc = baseDesc.clone();
        multiValueDesc.setDependencies([['d', 'point']]);
        multiValueDesc.setComputeAllValues(function (vals, outArr, numGraphElements) {
            for (var i = 0; i < outArr.length; i++) {
                outArr[i] = vals[i] * 2;
            }
        });

        ccManager.addComputedColumn(testDF, 'point', 'gSingle', singleValueDesc);
        ccManager.addComputedColumn(testDF, 'point', 'gMulti', multiValueDesc);

        var expectedArray = [0,0,2,2,4,4,6,6,8,8];
        var expectedCell = [4,4];

        expect(testDF.getCell(2, 'point', 'gSingle')).toEqual(expectedCell);
        expect(testDF.getColumnValues('gSingle', 'point')).toEqual(expectedArray);
        expect(testDF.getCell(2, 'point', 'gSingle')).toEqual(expectedCell);

        expect(testDF.getCell(2, 'point', 'gMulti')).toEqual(expectedCell);
        expect(testDF.getColumnValues('gMulti', 'point')).toEqual(expectedArray);
        expect(testDF.getCell(2, 'point', 'gMulti')).toEqual(expectedCell);

    });


});
