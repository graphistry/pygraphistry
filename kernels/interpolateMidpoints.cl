#include "common.h"

__kernel void interpolateMidpoints(
    __global uint2* edges,      // Array of [source node, target node]
    __global float2* points,   // Current point positions
    uint numEdges,                       // Length of edges array
    uint numSplits,                // Number of splits/midpoints in each edge
    __global float2* outputMidPoints    // Positions of interpolated midpoints
    )
{
    const int gid = get_global_id(0);
    const int global_size = get_global_size(0);
    uint2 edge;
    uint srcIdx, dstIdx;
    float2 prevPoint;
    float2 nextPoint;
    for (int i = gid; i < numEdges; i += global_size) {
        edge = edges[i];
        uint srcIdx = edge.x;
        uint dstIdx = edge.y;
        float2 srcPoint = points[srcIdx];
        float2 dstPoint = points[dstIdx];
        float2 step = (dstPoint - srcPoint) / (numSplits + 1);
        for (uint midPointIdx = 0; midPointIdx < numSplits; midPointIdx++) {
          outputMidPoints[(i * numSplits) + midPointIdx] = srcPoint + (step * midPointIdx);
        }
    }
}
