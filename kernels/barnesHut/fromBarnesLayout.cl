#include "common.h"
#include "barnesHut/barnesHutCommon.h"

__kernel void from_barnes_layout(
        //GRAPH_PARAMS
        float scalingRatio, float gravity, unsigned int edgeWeightInfluence, unsigned int flags,
        // number of points
        unsigned int numPoints,
        const __global float2* outputPositions,
        __global float *x_cords,
        __global float *y_cords,
        __global float* mass,
        __global volatile int* blocked,
        __global volatile int* maxdepthd,
        unsigned int step_number
){

    debugonce("from barnes layout\n");

    size_t gid = get_global_id(0);
    size_t global_size = get_global_size(0);
    for (int i = gid; i < numPoints; i += global_size) {
        outputPositions[i].x = x_cords[i];
        outputPositions[i].y = y_cords[i];
    }
}
