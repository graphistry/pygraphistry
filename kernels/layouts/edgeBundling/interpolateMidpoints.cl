#include "common.h"

__kernel void interpolateMidpoints(
    __global uint2* edges,      // Array of [source node, target node]
    __global float2* points,   // Current point positions
    uint numEdges,                       // Length of edges array
    uint numSplits,                // Number of splits/midpoints in each edge
    __global float2* outputMidPoints    // Positions of interpolated midpoints
    )
{
    const float HEIGHT = 0.0f;
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
        if (HEIGHT < FLT_EPSILON) {
            float2 step = (dstPoint - srcPoint) / (float) (numSplits + 1);
            for (uint midPointIdx = 0; midPointIdx < numSplits; midPointIdx++) {
              outputMidPoints[(i * numSplits) + midPointIdx] = srcPoint + (step * (midPointIdx + 1));
            }
        } else {
            float edgeLength = distance(srcPoint, dstPoint);
            float height = HEIGHT * (edgeLength / 2);
            float2 edgeDirection = normalize(srcPoint - dstPoint);
            float2 orthDirection = (float2) (edgeDirection.y, -1.0f * edgeDirection.x);
            float radius = (pown((edgeLength / 2), 2) + pown(height, 2)) / (2 * height);
            float2 midPoint = (float2) ((srcPoint.x + dstPoint.x) / 2.0f, (srcPoint.y + dstPoint.y) / 2.0f);
            float2 centerPoint = midPoint + ((radius - height) * (-1.0f * orthDirection));
            float theta = asin((edgeLength / 2) / radius) * 2.0f;
            float thetaStep = -theta / (float) (numSplits + 1);
            float2 startRadius = srcPoint - centerPoint;
            for (uint midPointIdx = 0; midPointIdx < numSplits; midPointIdx++) {
                float curTheta = thetaStep * (midPointIdx + 1);
                outputMidPoints[(i * numSplits) + midPointIdx] = 
                    centerPoint + 
                    (float2) ((cos(curTheta) * startRadius.x) - (sin(curTheta) * startRadius.y), 
                    (sin(curTheta) * startRadius.x) + (cos(curTheta) * startRadius.y));
            }
        }
    }
}
