//#define DEBUG
#include "common.h"

__kernel void selectNodesInCircle (
    // Selection
    const float center_x,
    const float center_y,
    const float radius_squared,
    const __global float2* restrict positions,
          __global char* restrict mask
) {
    const uint i = (uint) get_global_id(0);
    const float2 node = positions[i];

    const float d_x = node.x - center_x;
    const float d_y = node.y - center_y;
    if (d_x * d_x + d_y * d_y < radius_squared) {
        debug2("Selecting node %u", i);
        mask[i] = 1;
    } else {
        mask[i] = 0;
    }
}
