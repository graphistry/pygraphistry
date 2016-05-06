//#define DEBUG
#include "common.h"

__kernel void moveNodesByIds (
    // Ids to use
    const __global uint* ids,
    // Displacement
    const float deltaX,
    const float deltaY,
    __global float2* restrict inputPositions
) {
    const uint gid = (uint) get_global_id(0);
    const uint i = ids[gid];
    const float2 node = inputPositions[i];
    inputPositions[i] = (float2) (node.x + deltaX, node.y + deltaY);
}
