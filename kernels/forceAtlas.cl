//#define DEBUG
#include "common.h"


#define REPULSION_OVERLAP 0.00000001f
#define DEFAULT_NODE_SIZE 0.000001f
#define EPSILON 1.0f // bound whether d(a,b) == 0
#define SPEED 0.001f

#define IS_PREVENT_OVERLAP(flags) (flags & 1)
#define IS_STRONG_GRAVITY(flags)  (flags & 2)
#define IS_DISSUADE_HUBS(flags)   (flags & 4)
#define IS_LIN_LOG(flags)         (flags & 8)

//#define NOGRAVITY
//#define NOREPULSION
//#define NOATTRACTION

float repulsionForce(float2 distVec, uint n1Degree, uint n2Degree,
                          float scalingRatio, bool preventOverlap);

float gravityForce(float gravity, uint n1Degree, float2 centerVec, bool strong);

float attractionForce(float2 distVec, float n1Size, float n2Size, uint n1Degree, float weight, 
                      bool preventOverlap, uint edgeInfluence, bool linLog, bool dissuadeHubs);


//repulse points and apply gravity
__kernel void forceAtlasPoints (
    //input

    //GRAPH_PARAMS
    float scalingRatio, float gravity, unsigned int edgeWeightInfluence, unsigned int flags,

    __local float2* tilePointsParam, //FIXME make nodecl accept local params
    __local uint* tilePoints2Param, //FIXME make nodecl accept local params
    __local uint* tilePoints3Param, //FIXME make nodecl accept local params
    unsigned int numPoints,
    const __global float2* inputPositions,
    float width,
    float height,
    unsigned int stepNumber,
    const __global uint* inDegrees,
    const __global uint* outDegrees,

    //output
    __global float2* outputPositions
) {
    const unsigned int n1Idx = (unsigned int) get_global_id(0);
    const unsigned int tileSize = (unsigned int) get_local_size(0);
    const unsigned int numTiles = (unsigned int) get_num_groups(0);
    unsigned int modulus = numTiles / TILES_PER_ITERATION; // tiles per iteration:

    TILEPOINTS_INLINE_DECL;
    TILEPOINTS2_INLINE_DECL;
    TILEPOINTS3_INLINE_DECL;


    float2 n1Pos = inputPositions[n1Idx];
    float2 n1D = (float2) (0.0f, 0.0f);
    uint n1Degree = inDegrees[n1Idx] + outDegrees[n1Idx];
    debug2("ForceAtlasPoint (sourceIdx:%d)\n", n1Idx);

    for(unsigned int tile = 0; tile < numTiles; tile++) {
        if (tile % modulus != stepNumber % modulus) {
            continue;
        }

        const unsigned int tileStart = (tile * tileSize);
        unsigned int thisTileSize =  tileStart + tileSize < numPoints ?
                                        tileSize : numPoints - tileStart;

        //block on fetching current tile
        event_t waitEvents[3];
        waitEvents[0] = async_work_group_copy(TILEPOINTS, inputPositions + tileStart, thisTileSize, 0);
        waitEvents[1] = async_work_group_copy(TILEPOINTS2, inDegrees + tileStart, thisTileSize, 0);
        waitEvents[2] = async_work_group_copy(TILEPOINTS3, outDegrees + tileStart, thisTileSize, 0);
        wait_group_events(3, waitEvents);

        //hint fetch of next tile
        prefetch(inputPositions + ((tile + 1) * tileSize), thisTileSize);
        prefetch(inDegrees + ((tile + 1) * tileSize), thisTileSize);
        prefetch(outDegrees + ((tile + 1) * tileSize), thisTileSize);

        for(unsigned int cachedPoint = 0; cachedPoint < thisTileSize; cachedPoint++) {
            // Don't calculate the forces of a point on itself
            if (tileStart + cachedPoint == n1Idx) {
                continue;
            }

            float2 n2Pos = TILEPOINTS[cachedPoint];
            uint n2Degree = TILEPOINTS2[cachedPoint] + TILEPOINTS3[cachedPoint];
            float2 distVec = n1Pos - n2Pos;
            float rForce = repulsionForce(distVec, n1Degree, n2Degree, scalingRatio, 
                                          IS_PREVENT_OVERLAP(flags));

            debug4("\trForce (%d<->%d) %f\n", n1Idx, tileStart + cachedPoint, rForce);
            n1D += normalize(distVec) * rForce;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float2 dimensions = (float2) (width, height);
    const float2 centerVec = (dimensions / 2.0f) - n1Pos;
    float gForce = gravityForce(gravity, n1Degree, centerVec, IS_STRONG_GRAVITY(flags));
    debug3("gForce (%d) %f\n", n1Idx, gForce);

    outputPositions[n1Idx] =
        n1Pos
        + SPEED * normalize(centerVec) * gForce
        + SPEED * n1D;

    return;
}

#ifdef NOREPULSION
float repulsionForce(float2 distVec, uint n1Degree, uint n2Degree,
                          float scalingRatio, bool preventOverlap) {
    return 0.0f;
}
#else
float repulsionForce(float2 distVec, uint n1Degree, uint n2Degree,
                          float scalingRatio, bool preventOverlap) {
    float dist = length(distVec);
    int degreeProd = (n1Degree + 1) * (n2Degree + 1);
    float force;

    if (preventOverlap) {
        //FIXME include in prefetch etc, use actual sizes
        float n1Size = DEFAULT_NODE_SIZE;
        float n2Size = DEFAULT_NODE_SIZE;
        float distB2B = dist - n1Size - n2Size; //border-to-border

        force = distB2B > EPSILON  ? (scalingRatio * degreeProd / dist)
              : distB2B < -EPSILON ? (REPULSION_OVERLAP * degreeProd)
              : 0.0f;
    } else {
        force = scalingRatio * degreeProd / dist;
    }

    return clamp(force, 0.0f, 10.0f / SPEED);
}
#endif

#ifdef NOGRAVITY
float gravityForce(float gravity, uint n1Degree, float2 centerVec, bool strong) {
    return 0.0f;
}
#else
float gravityForce(float gravity, uint n1Degree, float2 centerVec, bool strong) {
    return gravity *
           (n1Degree + 1.0f) *
           (strong ? length(centerVec) : 1.0f);
}
#endif

//attract edges and apply forces
__kernel void forceAtlasEdges(
    //input

    //GRAPH_PARAMS
    float scalingRatio, float gravity, unsigned int edgeWeightInfluence, unsigned int flags,

    const __global uint2* springs,          // Array of springs, of the form [source node, target node] (read-only)
    const __global uint2* workList,             // Array of spring [source index, sinks length] pairs to compute (read-only)
    const __global float2* inputPoints,         // Current point positions (read-only)
    unsigned int stepNumber,

    //output
    __global float2* outputPoints
) {

    const size_t workItem = (unsigned int) get_global_id(0);
    const uint springsStart = workList[workItem].x;
    const uint springsCount = workList[workItem].y;
    const uint sourceIdx = springs[springsStart].x;
    debug2("ForceAtlasEdge (sourceIdx: %d)\n", sourceIdx);

    float2 n1Pos = inputPoints[sourceIdx];
    float n1Size = DEFAULT_NODE_SIZE; //FIXME include in prefetch etc, use actual sizes

    //FIXME start with previous deriv?
    float2 n1D = (float2) (0.0f, 0.0f);

    for(uint curSpringIdx = springsStart; curSpringIdx < springsStart + springsCount; curSpringIdx++) {
        const uint2 curSpring = springs[curSpringIdx];
        float2 n2Pos = inputPoints[curSpring.y];
        float n2Size = DEFAULT_NODE_SIZE;
        float2 distVec = n2Pos - n1Pos;

        float aForce = attractionForce(distVec, n1Size, n2Size, springsCount, 1.0f, 
                                       IS_PREVENT_OVERLAP(flags), edgeWeightInfluence, 
                                       IS_LIN_LOG(flags), IS_DISSUADE_HUBS(flags));
        debug4("\taForce (%d<->%d): %f\n", sourceIdx, curSpring.y, aForce);
        n1D += normalize(distVec) * aForce;
    }

    outputPoints[sourceIdx] = n1Pos + SPEED * n1D;
    return;
}

#ifdef NOATTRACTION
float attractionForce(float2 distVec, float n1Size, float n2Size, uint n1Degree, float weight, 
                      bool preventOverlap, uint edgeInfluence, bool linLog, bool dissuadeHubs) {
    return 0.0f;
}
#else
float attractionForce(float2 distVec, float n1Size, float n2Size, uint n1Degree, float weight, 
                      bool preventOverlap, uint edgeInfluence, bool linLog, bool dissuadeHubs) {

    float weightMultiplier = edgeInfluence == 0 ? 1.0f
                           : edgeInfluence == 1 ? weight
                                                : pown(weight, edgeInfluence);

    float dOffset = preventOverlap ? n1Size + n2Size : 0.0f;
    float dist = length(distVec) - dOffset;

    float aForce;
    if (preventOverlap && dist < EPSILON) {
        aForce = 0.0f;
    } else {
        float distFactor = (linLog ? log(1.0f + dist) : dist);
        float n1Deg = (dissuadeHubs ? n1Degree + 1.0f : 1.0f);
        aForce = weightMultiplier * distFactor / n1Deg;
    }

    return aForce * 10.0;
}
#endif
