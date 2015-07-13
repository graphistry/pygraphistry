#include "common.h"
#include "forceAtlas2/forceAtlas2Common.h"

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

    traction = isnan(traction) ? 0.0f : traction;
    swing = isnan(swing) ? 0.0f : swing;

    debug4("Swing/Traction (%d) %f\t%f\n", n1Idx, swing, traction);

    swings[n1Idx] = swing;
    tractions[n1Idx] = traction;

    return;
}
