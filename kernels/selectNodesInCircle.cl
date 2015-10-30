//#define DEBUG
#include "common.h"

__kernel void selectNodesInCircle (
    // Selection
    const float center_x,
    const float center_y,
    const float radius,
    const __global float2* restrict positions,
          __global char* restrict mask
) {
    const uint i = (uint) get_global_id(0);
    const float2 node = positions[i];
    const float2 radius_squared = radius * radius;
    coast float bottom = center_y - radius;
    const float top = center_y + radius;
    const float left = center_x - radius;
    const float right = center_x + radius;

    if (node.x > left && node.x < right &&
        node.y > bottom && node.y < top) {
        const float d_x = node.x - center_x;
        const float d_y = node.y - center_y;
        if (d_x * d_x + d_y * d_y < radius_squared) {
            debug2("Selecting node %u", i);
            mask[i] = 1;
        } else {
            mask[i] = 0;
        }
    } else {
        mask[i] = 0;
    }
}
