#include "common.h"

__kernel void faIntegrate (
    //input
    __global float* globalSpeed,
    const __global float2* inputPositions,
    const __global float2* curForces,
    const __global float* swings,
    //output
    __global float2* outputPositions
) {

    const unsigned int n1Idx = (unsigned int) get_global_id(0);
    const unsigned int numPoints = (unsigned int) get_global_size(0);

    const float speedFactor = 0.1f;
    const float maxSpeedFactor = 10.0f;

    float speed = speedFactor / (0.001f + sqrt(swings[n1Idx]));
    float maxSpeed = maxSpeedFactor / length(curForces[n1Idx]);

    float2 delta = *globalSpeed * min(maxSpeed, speed) * curForces[n1Idx];
    debug4("Speed (%d) %f max: %f\n", n1Idx, speed, maxSpeed);
    debug4("Delta (%d) %f\t%f\n", n1Idx, delta.x, delta.y);

    outputPositions[n1Idx] = inputPositions[n1Idx] + delta;
    return;
}
