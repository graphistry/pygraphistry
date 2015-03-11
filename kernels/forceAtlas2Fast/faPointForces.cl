#include "common.h"
#include "forceAtlas2Fast/forceAtlas2FastCommon.h"

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
