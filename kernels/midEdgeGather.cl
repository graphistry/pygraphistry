#include "common.h"

__kernel void midEdgeGather(
    const __global uint2* restrict edges,      // Array of [source node, target node]
    const __global float2* restrict midPoints, // Current midPoint positions (read-only)
    const __global float2* restrict points,   // Current point positions
    const uint numEdges,                       // Length of springs array.
    const uint numSplits,
    __global float4* restrict midEdgePositions    // Positions of the springs after forces are applied. 
                                                 // Length = len(springs) * 2: one float2 for start, 
                                                 // one float2 for end. (write-only)
    )
{
    const int gid = get_global_id(0);
    const int global_size = get_global_size(0);
    uint2 edge;
    int srcIdx, dstIdx;
    float2 prevPoint;
    float2 nextPoint;

    for (int i = gid; i < numEdges; i += global_size) {
        edge = edges[i];
        srcIdx = edge.x;
        dstIdx = edge.y;
        prevPoint = points[srcIdx];
        for (uint midPointIdx = 0; midPointIdx < numSplits; midPointIdx++) {
            nextPoint = midPoints[(i * numSplits) + midPointIdx];
            midEdgePositions[(i * (numSplits + 1)) + midPointIdx] = (float4) (prevPoint.x, prevPoint.y, nextPoint.x, nextPoint.y);
            debug5("Mid edge pos %f %f  ->  %f %f\n", prevPoint.x, prevPoint.y, nextPoint.x, nextPoint.y);
            debug3("Midpoint Index %u, midEdge index %u\n", (i * numSplits) + midPointIdx, (i * (numSplits + 1)) + midPointIdx);
            prevPoint = nextPoint;
        }
        midEdgePositions[(i + 1) * (numSplits + 1) - 1] = (float4) (prevPoint.x, prevPoint.y, points[dstIdx].x, points[dstIdx].y);
    }
}
