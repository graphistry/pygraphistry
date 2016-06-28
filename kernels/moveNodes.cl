//#define DEBUG
#include "common.h"

__kernel void moveNodes (
    // Selection
    const float top,
    const float left,
    const float bottom,
    const float right,
    // Displacement
    const float deltaX,
    const float deltaY,
    const __global float2* restrict inputPositions,
          __global float2* restrict outputPositions
) {
    const uint i = (uint) get_global_id(0);
    const float2 node = inputPositions[i];

    if (node.x > left && node.x < right &&
        node.y > bottom && node.y < top) {
        debug2("Moving node %u", i);
        outputPositions[i] = (float2) (node.x + deltaX, node.y + deltaY);
    } else {
        outputPositions[i] = node;
    }
}
