#include "common.h"

__kernel void midEdgeGather(
    const __global uint2* edges,      // Array of [source node, target node]
    const __global float2* restrict points,   // Current point positions
    const uint numEdges,                       // Length of edges array
    const uint numSplits,                // Number of splits/midpoints in each edge
    const __global float2* outputMidPoints    // Positions of midpoints after
                                              // interpolation
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
        srcIdx = edge.x;
        dstIdx = edge.y;
        srcPoint = points[srcIdx];
        dstPoint = points[dstIdx];
        float2 step = (dstPoint - srcPoint) / (numSplits + 1);
        /*xStep = (dstPoint.x - srcPoint.y) / (numSplits + 1);*/
        /*yStep = (dstPoint.y - srcPoint.y) / (numSplits + 1);*/
        for (uint midPointIdx = 0; midPointIdx < numSplits; midPointIdx++) {
          outputMidPoints[(i * numSplits) + midPointIdx] = srcPoint + (step * midPointIdx);
        }
    }
}
