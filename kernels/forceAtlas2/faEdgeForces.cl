#include "common.h"
#include "forceAtlas2/forceAtlas2Common.h"


__kernel void faEdgeForces(
    //input

    //GRAPH_PARAMS
    const float scalingRatio, const float gravity,
    const uint edgeInfluence, const uint flags,

    const __global uint2* edges,          // Array of springs, of the form [source node, target node] (read-only)
    const __global uint4* workList,             // Array of spring [edge index, sinks length, source index] triples to compute (read-only)
    const __global float2* inputPoints,         // Current point positions (read-only)
    const __global float2* partialForces,         // Current point positions (read-only)
    const uint stepNumber,
    const uint numWorkItems,

    //output
    __global float2* outputForces
) {

    int gid = get_global_id(0);
    int global_size = get_global_size(0);
    for (int workItem = gid; workItem < numWorkItems; workItem += global_size) {

        const uint springsStart = workList[workItem].x;
        const uint springsCount = workList[workItem].y;
        const uint nodeId       = workList[workItem].z;
        debug5("workList(%lu) = (%u, %u, %u)\n", workItem, springsStart, springsCount, nodeId);

        if (springsCount == 0) {
            debug2("No edge for sourceIdx: %u\n", nodeId);
            outputForces[nodeId] = partialForces[nodeId];
            continue;
        }

        const uint sourceIdx = edges[springsStart].x;
        debug2("ForceAtlasEdge (sourceIdx: %u)\n", sourceIdx);

        const float2 n1Pos = inputPoints[sourceIdx];
        const float n1Size = DEFAULT_NODE_SIZE; //FIXME include in prefetch etc, use actual sizes

        float2 n1D = (float2) (0.0f, 0.0f);

        for(uint curSpringIdx = springsStart; curSpringIdx < springsStart + springsCount; curSpringIdx++) {
            const uint2 curSpring = edges[curSpringIdx];
            const float2 n2Pos = inputPoints[curSpring.y];
            const float n2Size = DEFAULT_NODE_SIZE;
            const float2 distVec = n2Pos - n1Pos;

            const float aForce = attractionForce(distVec, n1Size, n2Size, springsCount, 1.0f,
                                                 IS_PREVENT_OVERLAP(flags), edgeInfluence,
                                                 IS_LIN_LOG(flags), IS_DISSUADE_HUBS(flags));
            debug4("\taForce (%d->%d): %f\n", sourceIdx, curSpring.y, aForce);
            n1D += normalize(distVec) * aForce;
        }

        debug4("PartialForce (%d) %f\t%f\n", sourceIdx, partialForces[sourceIdx].x, partialForces[sourceIdx].y);
        outputForces[sourceIdx] = partialForces[sourceIdx] + n1D;

    }

    return;

}
