//#define DEBUG
#include "common.h"

#define VT 16
#define WARPSIZE 32
#define THREADS 256
#define CARRYOUT_GLOBAL_MAX_SIZE 4096


// Segmented Reduction kernel, written for float2 types and addition operator
//
// Example of behavior (in 1D uint)
//
// numInput = 11
// numOutput = 4
// input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

//      indices: 0  1  2  3
// offsets =    [0, 3, 7, 9]
// segStart = [0,0,0,1,1,1,2,2,2,2,...]
//
// output = [6, 24, 17, 21]

__kernel void segReduce(
    const float scalingRatio, const float gravity,
    const uint edgeInfluence, const uint flags,
        uint numInput,
        __global float2* input,             // length = numInput
        __global uint2* edgeStartEndIdxs,
        __global uint* segStart,
        const __global uint4* workList,             // Array of spring [edge index, sinks length, source index] triples to compute (read-only)
        uint numOutput,
        __global float2* carryOut_global,    // length = ceil(numInput / local_size) capped at CARRYOUT_GLOBAL_MAX_SIZE
        __global float2* output,             // length = numOutput
        __global float2* partialForces
) {


    int gid = get_global_id(0);
    int global_size = get_global_size(0);
    int local_size = get_local_size(0);
    int tid = get_local_id(0);
    int group_id = get_group_id(0);

    const float2 ZERO = (float2) (0.0f, 0.0f);

    // SETUP FOR TESTING. DISREGARD LATER
    // 819.2 -> 820 output when offsets are 5

    // All outputs are 10,10 except for i=819 where it is 0,0

    /*for (int i = gid; i < numInput; i += global_size) {*/
        /*float val = 1.0f * (i % 5);*/
        /*input[i] = (float2) (val, val);*/
    /*}*/

    /*for (int i = gid; i < numOutput; i += global_size) {*/
        /*workList[i].z = i * 5;*/
        /*output[i] = (float2) (-1.0f, -1.0f);*/
    /*}*/

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);


     /*Naive implementation*/
     /*Limitted by one thread that must handle the longest segment*/
     /*Tested*/
     for (int i = gid; i < numOutput; i += global_size) {
         int start = edgeStartEndIdxs[i].x;
         int stop = edgeStartEndIdxs[i].y; //(i == numOutput - 1) ? numInput : workList[i + 1].x; // Exclusive stop -- should not be included in sum
         float2 accumulator = ZERO;

         for (int idx = start; idx < stop; idx++) {
             accumulator = accumulator + input[idx];
         }

         output[i] = accumulator + partialForces[i];
         /*return;*/
     }
     return;




    /*// TODO: Make sure we correctly round robin so we don't over/under do*/
    int initialOffset = VT * gid;
    int tidDelta = 0; // Measured in terms of threads (VT items per thread), not items
    float2 accumulator = ZERO;
    float2 localScan[VT];


    /*//////////////////////////////////////////////////////////////////////////*/
    /*// Preprocess offsets into rows?*/
    /*//////////////////////////////////////////////////////////////////////////*/

    // TODO: Have this done outside, since constant between ticks
    for (int i = gid; i < numOutput; i += global_size) {
        int start = edgeStartEndIdxs[i].x; //offsets[i];
        int stop = edgeStartEndIdxs[i].y; //3et3(i == numOutput - 1) ? numInput : offsets[i + 1]; // Exclusive stop -- should not be included in sum
        for (int idx = start; idx < stop; idx++) {
            segStart[idx] = i;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    /*//////////////////////////////////////////////////////////////////////////*/
    /*// Run the segmented scan inside the thread*/
    /*//*/
    /*// Tested*/
    /*//////////////////////////////////////////////////////////////////////////*/

    for (int i = 0; (i < VT) && (i + initialOffset < numInput); i++) {
            accumulator = i ? accumulator + input[initialOffset + i] : input[initialOffset + i]; // don't add if first
            localScan[i] = accumulator;
            if ((initialOffset + i + 1 == numInput) || (segStart[initialOffset + i]) != segStart[initialOffset + i + 1])
                accumulator = ZERO; // 0 for sum, 1 for mult
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    /*//////////////////////////////////////////////////////////////////////////*/
    /*// Figure out tidDelta*/
    /*//*/
    /*// tidDelta is the difference between a thread's index and the index*/
    /*// of the left-most thread ending in the same segment.*/
    /*// TODO: Optimize with spine scan and whatnot.*/
    /*//*/
    /*// Tested when all are 1. TODO: TEST MORE*/
    /*//////////////////////////////////////////////////////////////////////////*/

    int done = 0;
    int prevIdx = (gid - 1) * VT;
    int myIdx = gid * VT;
    while (!done && prevIdx >= 0 && myIdx + VT - 1 < numInput) {
        if (segStart[prevIdx + VT - 1] == segStart[myIdx + VT - 1]) {
            tidDelta += 1;
        } else {
            done = 1;
        }
        prevIdx -= VT;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    /*//////////////////////////////////////////////////////////////////////////*/
    /*// Run a parallel segmented scan over the carry-out values to compute*/
    /*// carry-in.*/
    /*//*/
    /*// This is a scan inside the work group, not between.*/
    /*//*/
    /*//////////////////////////////////////////////////////////////////////////*/

    float2 carryOut;
    float2 carryIn;
    /*// TODO: Make sure this is initialized to 0s*/
    __local float2 segScanBuffer[3*THREADS + 1]; // + 1 to be safe

    /*// Run an inclusive scan*/
    int first = 0;

    /*// This is the reduction of the last segment that each thread*/
    /*// is responsible for -- computed from earlier.*/
    segScanBuffer[tid + THREADS] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    for (int offset = 1; offset < THREADS; offset += offset) {

        if (tidDelta >= offset) {
            accumulator = segScanBuffer[first + tid - offset + THREADS] + accumulator;
        }
        first = THREADS - first; // alternates between 0 and THREADS
        segScanBuffer[first + tid + THREADS] = accumulator;
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

    /*// Get the exclusive scan.*/
    accumulator = tid ? segScanBuffer[first + tid - 1 + THREADS] : ZERO; // All but first thread have carryin
    carryOut = segScanBuffer[first + THREADS + THREADS - 1];
    carryIn = accumulator;

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    /*//////////////////////////////////////////////////////////////////////////*/
    /*// Store the carry-out for the entire workgroup to global memory.*/
    /*//////////////////////////////////////////////////////////////////////////*/

    /*// If first thread in work group, and group_id < size of array*/
    if (!tid && group_id < CARRYOUT_GLOBAL_MAX_SIZE)
        carryOut_global[group_id] = carryOut;

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    /*//////////////////////////////////////////////////////////////////////////*/
    /*// Add carry-in to each thread-local scan value. Store directly*/
    /*// to global.*/
    /*//////////////////////////////////////////////////////////////////////////*/

    /*// Pull in carryOut from previous workgroup if you're the first thread*/
    if (!tid && group_id > 0 && group_id < CARRYOUT_GLOBAL_MAX_SIZE) {
        carryIn += carryOut_global[group_id - 1];
    }

    for (int i = 0; (i < VT) && (i + initialOffset < numInput); i++) {
        // Add the carry-in to the local scan.
        float2 accumulator2 = carryIn + localScan[i];

        // Store on the end flag and clear the carry-in.
        if ((initialOffset + i + 1 == numInput) || (segStart[initialOffset + i] != segStart[initialOffset + i + 1])) {
            carryIn = ZERO;
            output[segStart[initialOffset + i]] = accumulator2;
        }
    }


    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    /*if (gid == 0) {*/
        /*printf("Expecting only one non-10 output\n");*/
        /*for (int i = 0; i < numOutput; i++) {*/
            /*if (!(output[i].x < 10.5f) || !(output[i].y > 9.5f)) {*/
                /*printf("output[%d] = %f, %f\n", i, output[i].x, output[i].y);*/
            /*}*/
        /*}*/
    /*}*/

    return;
}
