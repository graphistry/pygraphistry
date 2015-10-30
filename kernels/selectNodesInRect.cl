//#define DEBUG
#include "common.h"

__kernel void selectNodesInRect (
    // Selection
    const float top,
    const float left,
    const float bottom,
    const float right,
    const __global float2* restrict positions,
          __global char* restrict mask
) {
    const uint i = (uint) get_global_id(0);
    const float2 node = positions[i];

    if (node.x > left && node.x < right &&
        node.y > bottom && node.y < top) {
        debug2("Selecting node %u", i);
        mask[i] = 1;
    } else {
        mask[i] = 0;
    }
}
