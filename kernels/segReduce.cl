//#define DEBUG
#include "common.h"

#define VT 12
#define WARPSIZE 32
#define THREADS 256


// Segmented Reduction kernel, written for float2 types and addition operator
//
// Example of behavior (in 1D uint)
//
// numInput = 11
// numOutput = 4
// input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
// offsets = [0, 3, 7, 9]
//
// output = [6, 24, 17, 21]

__kernel void segReduce(
        uint numInput,
        __global float2* input,             // length = numInput
        __global uint* segStart,            // length = numInput
        __global uint* offsets,             // length = numOutput
        uint numOuput,
        __global float2* carryOut_global    // length = ceil(numInput / local_size)
        __global float2* output             // length = numOutput
) {


    int gid = get_global_id(0);
    int global_size = get_global_size(0);
    int local_size = get_local_size(0);
    int tid = get_local_id(0);
    int group_id = get_group_id(0);

    const float2 ZERO = (float2) (0.0f, 0.0f);

    // Naive implementation
    // Limitted by one thread that must handle the longest segment

    // for (int i = 0; i < numOutput; i += global_size) {
    //     int start = offsets[i];
    //     int stop = (i == numOutput - 1) ? numInput : offsets[i + 1]; // Exclusive stop -- should not be included in sum
    //     float2 accumulator = ZERO;

    //     for (int idx = start; idx < stop; idx++) {
    //         accumulator = accumulator + input[idx];
    //     }

    //     output[i] = accumulator;
    // }
    // return;


    // TODO: Make sure we correctly round robin so we don't over/under do
    int initialOffset = VT * gid;
    int tidDelta = 0; // Measured in terms of threads (VT items per thread), not items
    float2 accumulator;
    float2 localScan[VT];

    global int endFlags[], offsetArr[], data[], segStart[];


    //////////////////////////////////////////////////////////////////////////
    // Preprocess offsets into rows?
    //////////////////////////////////////////////////////////////////////////

    // TODO: Have this done outside, since constant between ticks
    for (int i = gid; i < numOutput; i += global_size) {
        int start = offsets[i];
        int stop = (i == numOutput - 1) ? numInput : offsets[i + 1]; // Exclusive stop -- should not be included in sum
        for (int idx = start; i < stop; i++) {
            segStart[idx] = start;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    //////////////////////////////////////////////////////////////////////////
    // Run the segmented scan inside the thread
    //////////////////////////////////////////////////////////////////////////

    for (int i = 0; i < VT && i + initialOffset < numInput; i++) {
        accumulator = i ? accumulator + data[initialOffset + i] : data[initialOffset + i]; // don't add if first
        localScan[initialOffset + i] = accumulator;
        if (segStart[initialOffset + i]) != segStart[initialOffset + i + 1]) accumulator = ZERO; // 0 for sum, 1 for mult
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    //////////////////////////////////////////////////////////////////////////
    // Figure out tidDelta
    //
    // tidDelta is the difference between a thread's index and the index
    // of the left-most thread ending in the same segment.
    // TODO: Optimize with spine scan and whatnot.
    //////////////////////////////////////////////////////////////////////////

    int done = 0;
    int prevIdx = (gid - 1) * VT;
    int myIdx = gid * VT;
    while (!done && prevIdx >= 0) {
        if (segStart[prevIdx + VT - 1] == segStart[myIdx]) {
            tidDelta += 1;
        } else {
            done = 1;
        }
        prevIdx -= VT;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    //////////////////////////////////////////////////////////////////////////
    // Run a parallel segmented scan over the carry-out values to compute
    // carry-in.
    //
    // This is a scan inside the work group, not between.
    //
    //////////////////////////////////////////////////////////////////////////

    float2 carryOut;
    float2 carryIn;
    local float2 segScanBuffer[THREADS + 1]; // + 1 to be safe

    // Run an inclusive scan
    int first = 0;

    // This is the reduction of the last segment that each thread
    // is responsible for -- computed from earlier.
    segScanBuffer[tid] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = 1; offset < THREADS; offset += offset) {
        if (tidDelta >= offset)
            accumulator = segScanBuffer[first + tid - offset] + accumulator;
        first = THREADS - first; // alternates between 0 and THREADS. TODO: Why?
        segScanBuffer[first + tid] = accumulator;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Get the exclusive scan.
    accumulator = tid ? segScanBuffer[first + tid - 1] : ZERO; // only first thread in workgroup has carryIn.
    carryOut = segScanBuffer[first + THREADS - 1];
    carryIn = accumulator;

    //////////////////////////////////////////////////////////////////////////
    // Store the carry-out for the entire workgroup to global memory.
    //////////////////////////////////////////////////////////////////////////

    if (!tid) carryOut_global[group_id] = carryOut;

    barrier(CLK_LOCAL_MEM_FENCE);
    //////////////////////////////////////////////////////////////////////////
    // Add carry-in to each thread-local scan value. Store directly
    // to global.
    //////////////////////////////////////////////////////////////////////////

    for (int i = 0; i < VT; i++) {
        // Add the carry-in to the local scan.
        float2 accumulator2 = carryIn + localScan[i];

        // Store on the end flag and clear the carry-in.
        if (rows[initialOffset + i] != rows[initialOffset + i + 1]) {
            carryIn = ZERO;
            output[rows[initialOffset + i]] = accumulator2;
        }
    }

    return;
}


