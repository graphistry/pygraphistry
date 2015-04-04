#include "common.h"
#include "barnesHut/barnesHutCommon.h"

// Computes BarnesHut specific data.
__kernel void to_barnes_layout(
        //GRAPH_PARAMS
        float scalingRatio, float gravity, unsigned int edgeWeightInfluence, unsigned int flags,
        // number of points
        const uint numPoints,
        const __global float2* inputPositions,
        __global float *x_cords,
        __global float *y_cords,
        __global float* mass,
        __global volatile int* blocked,
        __global volatile int* maxdepthd,
        const __global uint* pointDegrees,
        const uint step_number,
        const uint midpoint_stride,
        const uint midpoints_per_edge
){
    debugonce("to barnes layout\n");
    size_t gid = get_global_id(0);
    size_t global_size = get_global_size(0);


    for (int i = gid; i < numPoints; i += global_size) {
        x_cords[i] = inputPositions[(i * midpoints_per_edge) + midpoint_stride].x;
        y_cords[i] = inputPositions[(i * midpoints_per_edge) + midpoint_stride].y;
        mass[i] = 1.0f;
    }
    if (gid == 0) {
        *maxdepthd = -1;
        *blocked = 0;
    }
}
