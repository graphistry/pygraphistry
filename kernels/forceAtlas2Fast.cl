//#define DEBUG
#include "common.h"

// Speed tuning parameters
#define KS    0.1f
#define KSMAX 10.0f

#define REPULSION_OVERLAP 0.00000001f
#define DEFAULT_NODE_SIZE 0.000001f
#define EPSILON 0.00001f // bound whether d(a,b) == 0

#define IS_DISSUADE_HUBS(flags)   (flags & 1)
#define IS_LIN_LOG(flags)         (flags & 2)

//#define NOGRAVITY
//#define NOREPULSION
//#define NOATTRACTION

float attractionForce(const float2 distVec, const float n1Size, const float n2Size,
                      const uint n1Degree, const float weight, const bool noOverlap,
                      const uint edgeInfluence, const bool linLog, const bool dissuadeHubs);


//repulse points and apply gravity
__kernel void faPointForces (
    // #Define: preventOverlap, strongGravity
    //GRAPH_PARAMS
    const float scalingRatio, const float gravity,
    const uint edgeInfluence,

    __local float2* tilePointsParam, //FIXME make nodecl accept local params
    __local uint* tilePoints2Param, //FIXME make nodecl accept local params
    const uint numPoints,
    const uint tilesPerIteration,
    const __global float2* restrict inputPositions,
    const float width,
    const float height,
    const uint stepNumber,
    const __global uint* restrict pointDegrees,

    //output
    __global float2* pointForces
) {
    const uint n1Idx = (unsigned int) get_global_id(0);
    const uint tileSize = (unsigned int) get_local_size(0);
    const uint numTiles = (unsigned int) get_num_groups(0);
    const uint modulus = numTiles / tilesPerIteration;

    TILEPOINTS_INLINE_DECL;
    TILEPOINTS2_INLINE_DECL;

    const float2 n1Pos = inputPositions[n1Idx];
    const uint n1Degree = pointDegrees[n1Idx];
    debug3("ForceAtlasPoint (sourceIdx:%d, Degree:%d)\n", n1Idx, n1Degree);

    float2 n1D = (float2) (0.0f, 0.0f);

    for(unsigned int tile = 0; tile < numTiles; tile++) {
        if (tile % modulus != stepNumber % modulus) {
            continue; // Trade Speed for correctness
        }

        const uint tileStart = (tile * tileSize);
        const int thisTileSize =  tileStart + tileSize < numPoints ?
                                  tileSize : numPoints - tileStart;

        //block on fetching current tile
        event_t waitEvents[2];
        waitEvents[0] = async_work_group_copy(TILEPOINTS, inputPositions + tileStart, thisTileSize, 0);
        waitEvents[1] = async_work_group_copy(TILEPOINTS2, pointDegrees + tileStart, thisTileSize, 0);
        wait_group_events(2, waitEvents);

        //hint fetch of next tile
        prefetch(inputPositions + ((tile + 1) * tileSize), thisTileSize);
        prefetch(pointDegrees + ((tile + 1) * tileSize), thisTileSize);

        for(unsigned int cachedPoint = 0; cachedPoint < thisTileSize; cachedPoint++) {
            // Don't calculate the forces of a point on itself
            if (tileStart + cachedPoint == n1Idx) {
                continue;
            }

            const float2 n2Pos = TILEPOINTS[cachedPoint];
            const uint n2Degree = TILEPOINTS2[cachedPoint];
            const float2 distVec = n1Pos - n2Pos;
            const float dist = fast_length(distVec);
            const int degreeProd = (n1Degree + 1) * (n2Degree + 1);

            float rForce;
            #ifdef preventOverlap
            {
                float n1Size = DEFAULT_NODE_SIZE;
                float n2Size = DEFAULT_NODE_SIZE;
                float distB2B = dist - n1Size - n2Size; //border-to-border

                rForce = distB2B > EPSILON  ? (scalingRatio * degreeProd / dist)
                                            : distB2B < -EPSILON ? (REPULSION_OVERLAP * degreeProd)
                                            : 0.0f;
            }
            #else
            {
                rForce = scalingRatio * degreeProd / dist;
            }
            #endif

            debug4("\trForce (%d<->%d) %f\n", n1Idx, tileStart + cachedPoint, rForce);
            n1D += fast_normalize(distVec) * clamp(rForce, 0.0f, 1000000.0f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float2 dimensions = (float2) (width, height);
    const float2 centerVec = (dimensions / 2.0f) - n1Pos;
    float gForce;
    #ifdef strongGravity
    {
        gForce = gravity * (n1Degree + 1.0f) * fast_length(centerVec);
    }
    #else
    {
        gForce = gravity * (n1Degree + 1.0f);
    }
    #endif
    debug3("gForce (%d) %f\n", n1Idx, gForce);

    pointForces[n1Idx] = fast_normalize(centerVec) * gForce + n1D;

    return;
}



//attract edges and apply forces
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



// Compute global speed
__kernel void faSwingsTractions (
    //input
    const __global float2* prevForces,
    const __global float2* curForces,
    //output
    __global float* swings,
    __global float* tractions
) {
    const unsigned int n1Idx = (unsigned int) get_global_id(0);
    float2 prevForce = prevForces[n1Idx];
    float2 curForce = curForces[n1Idx];

    debug4("Prev Forces (%d) %f\t%f\n", n1Idx, prevForce.x, prevForce.y);
    debug4("Cur Forces  (%d) %f\t%f\n", n1Idx, curForce.x, curForce.y);

    float swing = length(curForce - prevForce);
    float traction = length(curForce + prevForce) / 2.0f;

    debug4("Swing/Traction (%d) %f\t%f\n", n1Idx, swing, traction);

    swings[n1Idx] = swing;
    tractions[n1Idx] = traction;

    return;
}


// Apply forces
__kernel void faIntegrate (
    //input
    float gSpeed,
    const __global float2* inputPositions,
    const __global float2* curForces,
    const __global float* swings,
    //output
    __global float2* outputPositions
) {

    const unsigned int n1Idx = (unsigned int) get_global_id(0);

    float speed = KS * gSpeed / (1.0f + gSpeed * sqrt(swings[n1Idx]));
    float maxSpeed = KSMAX / length(curForces[n1Idx]);
    float2 delta = min(speed, maxSpeed) * curForces[n1Idx];

    debug4("Speed (%d) %f max: %f\n", n1Idx, speed, maxSpeed);
    debug4("Delta (%d) %f\t%f\n", n1Idx, delta.x, delta.y);

    outputPositions[n1Idx] = inputPositions[n1Idx] + delta;
    return;
}


// Apply forces and estimate global Speed
__kernel void faIntegrate2 (
    //input
    unsigned int numPoints,
    float tau,
    const __global float2* inputPositions,
    const __global uint* pointDegrees,
    const __global float2* curForces,
    const __global float* swings,
    const __global float* tractions,
    //output
    __global float2* outputPositions
) {
    const unsigned int n1Idx = (unsigned int) get_global_id(0);
    const unsigned int tileSize = (unsigned int) get_local_size(0);
    const unsigned int tile = (unsigned int) get_local_id(0);

    const uint tileStart = (tile * tileSize);
    const uint thisTileSize = tileStart + tileSize < numPoints ? tileSize : numPoints - tileStart;

    float gTraction = 0.0f;
    float gSwing = 0.0f;

    for(uint n2Idx = tileStart; n2Idx < tileStart + thisTileSize; n2Idx++) {
        const uint n2Degree = pointDegrees[n2Idx];
        gSwing += (n2Degree + 1) * swings[n2Idx];
        gTraction += (n2Degree + 1) * tractions[n2Idx];
    }

    float gSpeed = tau * gTraction / gSwing;
    debug3("Global Speed (tile: %d) %f\n", tile, gSpeed);

    float speed = KS * gSpeed / (1.0f + gSpeed * sqrt(swings[n1Idx]));
    float maxSpeed = KSMAX / length(curForces[n1Idx]);
    float2 delta = min(speed, maxSpeed) * curForces[n1Idx];

    debug4("Speed (%d) %f max: %f\n", n1Idx, speed, maxSpeed);
    debug4("Delta (%d) %f\t%f\n", n1Idx, delta.x, delta.y);

    outputPositions[n1Idx] = inputPositions[n1Idx] + delta;
    return;
}
