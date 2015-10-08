#include "common.h"
#include "forceAtlas2Fast/forceAtlas2FastCommon.h"


float attractionForce(const float2 distVec, const float n1Size, const float n2Size,
                      const uint n1Degree, const float weight, const bool noOverlap,
                      const uint edgeInfluence, const bool linLog, const bool dissuadeHubs) {

    const float weightMultiplier = edgeInfluence == 0 ? 1.0f
                                 : edgeInfluence == 1 ? weight
                                                      : pown(weight, edgeInfluence);

    const float dOffset = noOverlap ? n1Size + n2Size : 0.0f;
    const float dist = length(distVec) - dOffset;

    float aForce;
    if (noOverlap && dist < EPSILON) {
        aForce = 0.0f;
    } else {
        const float distFactor = (linLog ? log(1.0f + dist) : dist);
        const float n1Deg = (dissuadeHubs ? n1Degree + 1.0f : 1.0f);
        aForce = weightMultiplier * distFactor / n1Deg;
    }

#ifndef NOATTRACTION
    return aForce;
#else
    return 0.0f;
#endif
}


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

    //output
    __global float2* outputForces
) {

    const size_t workItem = (unsigned int) get_global_id(0);

    const uint springsStart = workList[workItem].x;
    const uint springsCount = workList[workItem].y;
    const uint nodeId       = workList[workItem].z;
    debug5("workList(%lu) = (%u, %u, %u)\n", workItem, springsStart, springsCount, nodeId);

    if (springsCount == 0) {
        debug2("No edge for sourceIdx: %u\n", nodeId);
        outputForces[nodeId] = partialForces[nodeId];
        return;
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

        float aForce;
        #ifdef preventOverlap
        {
            aForce = attractionForce(distVec, n1Size, n2Size, springsCount, 1.0f,
                                     true, edgeInfluence,
                                     IS_LIN_LOG(flags), IS_DISSUADE_HUBS(flags));
        }
        #else
        {
            aForce = attractionForce(distVec, n1Size, n2Size, springsCount, 1.0f,
                                     false, edgeInfluence,
                                     IS_LIN_LOG(flags), IS_DISSUADE_HUBS(flags));
        }
        #endif

        debug4("\taForce (%d->%d): %f\n", sourceIdx, curSpring.y, aForce);
        n1D += normalize(distVec) * aForce;
    }

    debug4("PartialForce (%d) %f\t%f\n", sourceIdx, partialForces[sourceIdx].x, partialForces[sourceIdx].y);
    outputForces[sourceIdx] = partialForces[sourceIdx] + n1D;
    return;
}
