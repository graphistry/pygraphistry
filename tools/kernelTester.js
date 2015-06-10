'use strict';

var Kernel      = require('../js/kernel.js');
var cljs        = require('../js/cl.js');
var _           = require('underscore');
var simCl       = require('../js/SimCL.js');
var RenderNull  = require('../js/RenderNull.js');

// Things to do:
//
// Initialize CL Context // DONE
// Cast over to buffer.
// Read back from buffer.


var KernelTester = function (fileName, clContext) {
    this.clContext = clContext;
    this.fileName = fileName;

    this.numWorkGroups = 1;
    this.workGroupSize = 256;
    this.argTypes = {};
    this.argNames = [];
    this.argValues = {};
};


KernelTester.prototype.exec = function () {
    var that = this;
    var kernel = new Kernel(this.fileName, this.argNames,
            this.argTypes, this.fileName, this.clContext);

    // Set arguments
    var setObj = {};
    _.each(this.argNames, function (name) {
        setObj[name] = this.argValues[name];
    });
    kernel.set(setObj);

    var startTime = process.hrtime();
    var diff = 0;
    kernel.exec([this.numWorkGroups], [], [this.workGroupSize])
        .then(function () {
            that.clContext.queue.finish();
            diff = process.hrtime(start);

            // Print out all relevant stuff.
            console.log('Finished executing ' + that.fileName + ' in ' + diff);
            // TODO: Read back and print buffers out.

        });
};

KernelTester.prototype.setNumWorkGroups = function (num) {
    this.numWorkGroups = num;
};

KernelTester.prototype.setWorkGroupSize = function (size) {
    this.workGroupSize = size;
};

// Null if buffer.
KernelTester.prototype.setArgTypes = function (argTypesObj) {
    _.each(_.keys(argTypesObj), function (key) {
        this.argTypes[key] = argTypesObj[key];
    });
};

KernelTester.prototype.setArgNames = function (argNameArray) {
    this.argNames = argNameArray;
};

// Must set arg values after arg types.
// Currently assumes all buffers exist as 32bit values.
KernelTester.prototype.setArgValues = function (argValuesObj) {
    _.each(_.keys(argValuesObj), function (key) {
        // TODO: Convert to appropriate type and load onto GPU/buffer.
        if (this.argTypes[key] !== null) {
            this.argValues[key] = argValuesObj[key];
        } else {
            var orig = argValuesObj[key];
            var newBuffer = this.clContext.createBuffer((orig.length) * Float32Array.BYTES_PER_ELEMENT, key);
            this.argValues[key] = newBuffer.buffer;
        }
    });
};

// __kernel void segReduce(
//     const float scalingRatio, const float gravity,
//     const uint edgeInfluence, const uint flags,
//         uint numInput,
//         __global float2* input,             // length = numInput
//         __global uint2* edgeStartEndIdxs,
//         __global uint* segStart,
//         const __global uint4* workList,             // Array of spring [edge index, sinks length, source index] triples to compute (read-only)
//         uint numOutput,
//         __global float2* carryOut_global,    // length = ceil(numInput / local_size) capped at CARRYOUT_GLOBAL_MAX_SIZE
//         __global float2* output,             // length = numOutput
//         __global float2* partialForces
// ) {

function makeZeroFilledArray(length) {
    return Array.apply(null, new Array(length)).map(Number.prototype.valueOf,0);
}

function mainTestFunction (clContext) {
    console.log('Starting main test function');

    var tester = new KernelTester('segReduce.cl', clContext);
    var argNames = ['scalingRatio', 'gravity', 'edgeInfluence', 'flags',
            'numInput', 'input', 'edgeStartEndIdxs', 'segStart', 'workList',
            'numOutput', 'carryOut_global', 'output', 'partialForces'
    ];

    var argTypes = {
        scalingRatio: cljs.types.float_t,
        gravity: cljs.types.float_t,
        edgeInfluence: cljs.types.uint_t,
        flags: cljs.types.uint_t,
        numInput: cljs.types.uint_t,
        input: null,
        edgeStartEndIdxs: null,
        segStart: null,
        workList: null,
        numOutput: cljs.types.uint_t,
        carryOut_global: null,
        output: null,
        partialForces: null
    };

    var numWorkGroups = 256;
    var workGroupSize = 256;

    // Values for arguments

    var input = [1,1,2,2,3,3,1,1,2,2,3,3] // Double length so 6
    var numInput = input.length / 2;
    var edgeStartEndIdxs = [0,1,1,4,4,5];
    var segStart = [0,1,4];
    var workList = [0];
    var numOutput = segStart.length;
    var carryOut_global = makeZeroFilledArray(10);
    var output = makeZeroFilledArray(numOutput);
    var partialForces = makeZeroFilledArray(numOutput);


    var argValues = {
        scalingRatio: 1.0,
        gravity: 1.0,
        edgeInfluence: 0.0,
        flags: 0,
        numInput: numInput,
        input: input,
        edgeStartEndIdxs: edgeStartEndIdxs,
        segStart: segStart,
        workList: workList,
        numOutput: numOutput,
        carryOut_global: carryOut_global,
        output: output,
        partialForces: partialForces
    };

    tester.setNumWorkGroups(numWorkGroups);
    tester.setWorkGroupSize(workGroupSize);
    tester.setArgNames(argNames);
    tester.setArgTypes(argTypes);
    tester.setArgValues(argValues);

    tester.exec();
}



if (require.main === module) {
    console.log('Running kernelTester in standalone mode.');

    var DEVICE = 'gpu';
    var VENDOR = 'default';

    // TODO: Get renderer
    RenderNull.create(null)
        .then(function (renderer) {
            // console.log('Got renderer');
            cljs.create(renderer, DEVICE, VENDOR)
                .then(function (clContext) {
                    // console.log('Got cl context');
                    mainTestFunction(clContext);
                });
        });
}

module.exports = KernelTester;
