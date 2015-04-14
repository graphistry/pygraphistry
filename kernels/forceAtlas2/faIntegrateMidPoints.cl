#include "common.h"
#include "forceAtlas2/forceAtlas2Common.h"

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

    #define SPEED_CONSTANT 5.0f

    float sqrtPoints = sqrt((float)numPoints);
    float speedFactor = max(SPEED_CONSTANT * sqrtPoints / 1000.0f, 0.1f);
    float maxSpeedFactor = max(SPEED_CONSTANT * sqrtPoints / 10.0f, 10.0f);


    float normalizedSwing = sqrt( (swings[n1Idx] ) / (sqrtPoints) );
    float speed = speedFactor * (*globalSpeed) / (1.0f + (*globalSpeed) * normalizedSwing);
    float maxSpeed = maxSpeedFactor / length(curForces[n1Idx]);


    /*float2 delta = min(speed, maxSpeed) * curForces[n1Idx];*/
    float2 delta = curForces[n1Idx] * min(speed, maxSpeed);

    debug4("Speed (%d) %f max: %f\n", n1Idx, speed, maxSpeed);
    debug4("Delta (%d) %f\t%f\n", n1Idx, delta.x, delta.y);

    /*outputPositions[n1Idx] = inputPositions[n1Idx] + delta;*/
    outputPositions[n1Idx] = inputPositions[n1Idx] + delta;
    return;
}
