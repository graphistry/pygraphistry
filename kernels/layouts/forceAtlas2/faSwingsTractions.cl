#include "common.h"

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

    // These should never be nan if prevForces is initialized properly and all curForces
    // are calculated correctly. This can be a useful check however.  
    /*traction = isnan(traction) ? 1.0f : traction;*/
    /*swing = isnan(swing) ? 1.0f : swing;*/

    swings[n1Idx] = (0.1f * swing) + 0.9f * swings[n1Idx];
    tractions[n1Idx] = (0.1f * traction) + 0.9f * tractions[n1Idx];

    return;
}
