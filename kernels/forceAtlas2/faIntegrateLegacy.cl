#include "common.h"
#include "forceAtlas2/forceAtlas2Common.h"

__kernel void faIntegrateLegacy (
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
