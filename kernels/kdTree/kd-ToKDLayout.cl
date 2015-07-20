/*#define DEBUG*/
#include "common.h"
#undef DEBUG
#include "barnesHut/barnesHutCommon.h"

// transforms buffers for more optimized memory accesses.
__kernel void to_kd_layout(
        // number of points
        const uint numPoints,
        const __global float2* inputMidPositions,
        const __global float2* inputPositions,
        __global float *x_cords,
        __global float *y_cords,
        const __global uint2* springs,
        __global float *edgeDirectionX,
        __global float *edgeDirectionY,
        __global float* edgeLengths,
        __global float* mass,
        __global volatile int* blocked,
        __global volatile int* maxdepthd,
        const uint step_number,
        const uint midpoint_stride,
        const uint midpoints_per_edge
){
    debugonce("to barnes layout\n");
    size_t gid = get_global_id(0);
    size_t global_size = get_global_size(0);


    if (gid == 0) {
    debug2("Num of points %u \n", numPoints);
    }

    uint src, target;
    float2 directionVector;
    float distanceVector;
    uint index;
    for (int i = gid; i < numPoints; i += global_size) {
        index = (i * midpoints_per_edge) + midpoint_stride;
        /*x_cords[i] = inputMidPositions[index].x;*/
        /*y_cords[i] = inputMidPositions[index].y;*/
        x_cords[i] = inputMidPositions[index].x * 1000.0f;
        y_cords[i] = inputMidPositions[index].y * 1000.0f;
        mass[i] = 1.0f;
        src = springs[i].x;
        target = springs[i].y;
        debug4("Target (%u), X: %f, Y %f \n", target, inputPositions[target].x, inputPositions[target].y);
        debug4("Src (%u), X: %f, Y %f \n", src, inputPositions[src].x, inputPositions[src].y);
        directionVector = (float2) normalize(inputPositions[target] - inputPositions[src]);
        distanceVector = (float) distance(inputPositions[target],inputPositions[src]);
        edgeDirectionX[i] = directionVector.x;
        edgeDirectionY[i] = directionVector.y;
        edgeLengths[i] = distanceVector;
        debug4("Edge direction (%u), X: %f, Y: %f \n", index, directionVector.x, directionVector.y);
    }
    if (gid == 0) {
        *maxdepthd = -1;
        *blocked = 0;
    }
}
