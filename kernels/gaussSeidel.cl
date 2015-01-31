//#define DEBUG
#include "common.h"
#include "gsCommon.cl"

// Calculate the force of point b on point a, returning a vector indicating the movement to point a
float2 pointForce(float2 a, float2 b, float force);


// Retrieves a random point from a set of points
float2 randomPoint(__local float2* points, unsigned int numPoints, __constant float2* randValues,
    unsigned int randOffset);


__kernel void gaussSeidelPoints(
    unsigned int numPoints,
    unsigned int tilesPerIteration,
    const __global float2* inputPositions,
    __global float2* outputPositions,
    __local float2* tilePointsParam, //FIXME make nodecl accept local params
    float width,
    float height,
    float charge,
    float gravity,
    __constant float2* randValues,
    unsigned int stepNumber)
{

    // use async_work_group_copy() and wait_group_events() to fetch the data from global to local
    // use vloadn() and vstoren() to read/write vectors.

    TILEPOINTS_INLINE_DECL;

    const float2 dimensions = (float2) (width, height);

    const float alpha = max(0.1f * pown(0.99f, floor(convert_float(stepNumber) / (float) tilesPerIteration)), 0.005f);  //1.0f / (clamp(((float) stepNumber), 1.0f, 50.0f) + 10.0f);

    const unsigned int pointId = (unsigned int) get_global_id(0);

    // The point we're updating
    float2 myPos = inputPositions[pointId];

    // Points per tile = threads per workgroup
    const unsigned int tileSize = (unsigned int) get_local_size(0);
    const unsigned int numTiles = (unsigned int) get_num_groups(0);

    float2 posDelta = (float2) (0.0f, 0.0f);

  unsigned int modulus = numTiles / tilesPerIteration; // tiles per iteration:


    for(unsigned int tile = 0; tile < numTiles; tile++) {

        if (tile % modulus != stepNumber % modulus) {
            continue;
        }

        const unsigned int tileStart = (tile * tileSize);

        // If numPoints isn't a multiple of tileSize, the last tile will have less than the full
        // number of points. If we detect we'd be reading out-of-bounds data, clamp the number of
        // points we read to be within bounds.
        unsigned int thisTileSize =  tileStart + tileSize < numPoints ?
                                        tileSize : numPoints - tileStart;


        // if(threadLocalId < thisTileSize){
        //  tilePoints[threadLocalId] = inputPositions[tileStart + threadLocalId];
        // }

        // barrier(CLK_LOCAL_MEM_FENCE);


        event_t waitEvents[1];

        //continue; //FIXME continuing loop from here busts code if tilePoints is dynamic param

        waitEvents[0] = async_work_group_copy(TILEPOINTS, inputPositions + tileStart, thisTileSize, 0);
        wait_group_events(1, waitEvents);
        prefetch(inputPositions + ((tile + 1) * tileSize), thisTileSize);


        for(unsigned int cachedPoint = 0; cachedPoint < thisTileSize; cachedPoint++) {
            // Don't calculate the forces of a point on itself
            if(tileStart + cachedPoint == pointId) {
                continue;
            }

            float2 otherPoint = TILEPOINTS[cachedPoint];

            // for(uchar tries = 0; fast_distance(otherPoint, myPos) <= FLT_EPSILON && tries < 100; tries++) {
            if(fast_distance(otherPoint, myPos) <= FLT_EPSILON) {
                otherPoint = randomPoint(TILEPOINTS, thisTileSize, randValues, stepNumber);
            }

            posDelta += pointForce(myPos, otherPoint, charge * alpha);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Force of gravity pulling the points toward the center
    float2 center = dimensions / 2.0f;
    // TODO: Should we be dividing the stength of gravity by tilesPerIteration? We only consider
    // 1 / tilesPerIteration of the total points in any execution, but here we apply full gravity.
    posDelta += ((float2) ((center.x - myPos.x), (center.y - myPos.y)) * (gravity * alpha));


    // Clamp myPos to be within the walls
    // outputPositions[pointId] = clamp(myPos + posDelta, (float2) (0.0f, 0.0f), dimensions);

    outputPositions[pointId] = myPos + posDelta;

    return;
}



//for each edge source, find corresponding point and tension from destination points
__kernel void gaussSeidelSprings(
    unsigned int tilesPerIteration,
    const __global uint2* springs,         // Array of springs, of the form [source node, target node] (read-only)
    const __global uint4* workList,            // Array of spring [source index, sinks length] pairs to compute (read-only)
    const __global uint* edgeTags,          // Array of worklist item -> 0/1
    const __global float2* inputPoints,      // Current point positions (read-only)
    __global float2* outputPoints,     // Point positions after spring forces have been applied (write-only)
    float springStrength0,              // The rigidity of the springs
    float springDistance0,              // The 'at rest' length of a spring
    float springStrength1,              // The rigidity of the springs
    float springDistance1,              // The 'at rest' length of a spring
    unsigned int stepNumber
    )
{
    const float alpha = max(0.1f * pown(0.99f, floor(convert_float(stepNumber) / (float) tilesPerIteration)), 0.005f);
    // const float alpha = max(0.1f * pown(0.99f, stepNumber), FLT_EPSILON * 2.0f);

    // From Hooke's Law, we generally have that the force exerted by a spring is given by
    //  F = -k * X, where X is the distance the spring has been displaced from it's natural
    // distance, and k is some constant positive real number.

    // d = target - source;
    // l1 = Math.sqrt(distance^2);
    // l = alpha * strengths[i] * ((l1) - distances[i]) / l1;
    // distance *= l;
    // k = source.weight / (target.weight + source.weight)
    // target -= distance * k;
    // k = 1 - k;
    // source += distance * k;

    const size_t workItem = (unsigned int) get_global_id(0);

    const uint springsStart = workList[workItem].x;
    const uint springsCount = workList[workItem].y;
    const uint nodeId = workList[workItem].z;

    if (springsCount == 0) {
        outputPoints[nodeId] = inputPoints[nodeId];
        return;
    }

    const uint sourceIdx = springs[springsStart].x;

    float2 source = inputPoints[sourceIdx];
    for(uint curSpringIdx = springsStart; curSpringIdx < springsStart + springsCount; curSpringIdx++) {
        const uint2 curSpring = springs[curSpringIdx];
        float2 target = inputPoints[curSpring.y];
        float dist = distance(target, source); //sqrt((delta.x * delta.x) + (delta.y * delta.y));
        if(dist > FLT_EPSILON) {
            uint edgeTag = edgeTags[curSpringIdx];
            float force = alpha
                * (edgeTag ? springStrength1 : springStrength0)
                * (dist - (edgeTag ? springDistance1 : springDistance0)) / dist;
            source += (target - source) * force;
        }
    }
    outputPoints[sourceIdx] = source;
}


__kernel void gaussSeidelSpringsGather(
    const __global uint2* springs,         // Array of springs, of the form [source node, target node] (read-only)
    const __global uint4* workList,            // Array of spring [source index, sinks length] pairs to compute (read-only)
    const __global float2* inputPoints,      // Current point positions (read-only)
    __global float4* springPositions   // Positions of the springs after forces are applied. Length
                                       // len(springs) * 2: one float2 for start, one float2 for
                                       // end. (write-only)
    )
{

    const size_t workItem = (unsigned int) get_global_id(0);
    const uint springsStart = workList[workItem].x;
    const uint springsCount = workList[workItem].y;
    if (springsCount == 0)
        return;

    const uint sourceIdx = springs[springsStart].x;
    const float2 source = inputPoints[sourceIdx];

    for (uint curSpringIdx = springsStart; curSpringIdx < springsStart + springsCount; curSpringIdx++) {
        const uint2 curSpring = springs[curSpringIdx];
        const float2 target = inputPoints[curSpring.y];
        debug5("Spring pos %f %f  ->  %f %f\n", source.x, source.y, target.x, target.y);
        springPositions[curSpringIdx] = (float4) (source.x, source.y, target.x, target.y);
    }

}
