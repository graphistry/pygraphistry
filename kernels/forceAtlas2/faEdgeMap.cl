#include "common.h"
#include "forceAtlas2/forceAtlas2Common.h"

//attract edges and apply forces
__kernel void faEdgeMap(
    //input

    //GRAPH_PARAMS
    const float scalingRatio, const float gravity,
    const uint edgeInfluence, const uint flags,

    const __global uint2* edges,          // Array of springs, of the form [source node, target node] (read-only)
    const uint numEdges,
    const __global uint4* workList,             // Array of spring [edge index, sinks length, source index] triples to compute (read-only)
    const __global float2* inputPoints,         // Current point positions (read-only)
    const __global float* edgeWeights,
    const uint stepNumber,
    const uint numWorkItems,
    //output
    __global float2* outputForcesMap
) {

    int gid = get_global_id(0);
    int global_size = get_global_size(0);
    for (int workItem = gid; workItem < numEdges; workItem += global_size) {
        const uint sourceIdx = edges[workItem].x;
        const uint targetIdx = edges[workItem].y;
        const float2 n1Pos = inputPoints[sourceIdx];
        const float2 n2Pos = inputPoints[targetIdx];
        const uint springsCount = 1; //workList[sourceIdx].y;// numEdges[workItem]; TODO (paden) make this work
        const float n2Size = DEFAULT_NODE_SIZE;
        const float n1Size = DEFAULT_NODE_SIZE;
        const float2 distVec = n2Pos - n1Pos;


        // TODO (paden) This can probably be optimized
        const float aForce = attractionForce(distVec, n1Size, n2Size, springsCount, edgeWeights[workItem],
                                             IS_PREVENT_OVERLAP(flags), edgeInfluence,
                                             IS_LIN_LOG(flags), IS_DISSUADE_HUBS(flags));

        const float2 n1D = normalize(distVec) * aForce;

        outputForcesMap[workItem] = n1D;

    }
    return;
}
