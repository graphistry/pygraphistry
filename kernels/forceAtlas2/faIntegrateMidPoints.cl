/*#define DEBUG*/
#include "common.h"
#undef DEBUG
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
    // Set to 0.1f
    /*float speedFactor = max(SPEED_CONSTANT * sqrtPoints / 1000.0f, 0.1f);*/
    float speedFactor = 0.05f;
    // Set to 10
    //
    /*float maxSpeedFactor = max(SPEED_CONSTANT * sqrtPoints / 10.0f, 10.0f);*/
    float maxSpeedFactor = 0.01f;


    float2 delta;
    if (swings[n1Idx] > 0.0f) {
    float temp = 10000.0f * swings[n1Idx];
    float normalizedSwing = pow((temp  / (sqrtPoints) ), 2.0f);
    float speed = speedFactor * (*globalSpeed) / (1.0f + (*globalSpeed) * normalizedSwing);
    float maxSpeed = maxSpeedFactor / length(curForces[n1Idx]);


    delta = min(speed, maxSpeed) * curForces[n1Idx];
    /*float2 delta = (float2) curForces[n1Idx]; */

    debug6("Speed (%d) %f max: %f, global_speed %f swing %.9g \n", n1Idx, speed, maxSpeed, *globalSpeed, normalizedSwing);
    /*debug6("Delta in integrate (%d) %f\t%f \nforces x %f, y %f\n", n1Idx, delta.x, delta.y, curForces[n1Idx].x, curForces[n1Idx].y);*/
    }  else {
      delta = (float2) (0.0f, 0.0f);
   }

    /*outputPositions[n1Idx] = inputPositions[n1Idx] + delta;*/
    outputPositions[n1Idx] = inputPositions[n1Idx] + delta;
    return;
}
