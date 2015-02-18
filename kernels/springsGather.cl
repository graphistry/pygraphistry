//#define DEBUG
#include "common.h"

__kernel void springsGather(
    const __global uint2* restrict springs,      // Array of [source node, target node]
    const __global float2* restrict inputPoints, // Current point positions (read-only)
    const uint numSprings,                       // Length of springs array.
    __global float4* restrict springPositions    // Positions of the springs after forces are applied. 
                                                 // Length = len(springs) * 2: one float2 for start, 
                                                 // one float2 for end. (write-only)
    )
{
    const int gid = get_global_id(0);
    const int global_size = get_global_size(0);
    uint2 spring;
    float2 src, dst;

    for (int i = gid; i < numSprings; i += global_size) {
        spring = springs[i];
        src = inputPoints[spring.x];
        dst = inputPoints[spring.y];
        debug5("Spring pos %f %f  ->  %f %f\n", src.x, src.y, dst.x, dst.y);
        springPositions[i] = (float4) (src.x, src.y, dst.x, dst.y);
    }
}
