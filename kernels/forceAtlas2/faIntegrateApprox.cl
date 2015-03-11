#include "common.h"
#include "forceAtlas2/forceAtlas2Common.h"

__kernel void faIntegrateApprox (
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