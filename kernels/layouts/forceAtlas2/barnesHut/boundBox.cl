#include "common.h"
#include "layouts/forceAtlas2/barnesHut/barnesHutCommon.h"

__kernel void bound_box(
        //graph params
        float scalingRatio, float gravity, unsigned int edgeWeightInfluence, unsigned int flags,
        __global float *x_cords,
        __global float *y_cords,
        __global float* accx,
        __global float* accy,
        __global int* children,
        __global float* mass,
        __global int* start,
        __global int* sort,
        __global float* global_x_mins,
        __global float* global_x_maxs,
        __global float* global_y_mins,
        __global float* global_y_maxs,
        __global float* global_swings,
        __global float* global_tractions,
        __global float* swings,
        __global float* tractions,
        __global int* count,
        __global volatile int* blocked,
        __global volatile int* stepd,
        __global volatile int* bottomd,
        __global volatile int* maxdepthd,
        __global volatile float* radiusd,
        __global volatile float* globalSpeed,
        unsigned int step_number,
        float width,
        float height,
        const int num_bodies,
        const int num_nodes,
        __global float2* pointForces,
        float tau
){

    size_t tid = get_local_id(0);
    size_t gid = get_group_id(0);
    size_t dim = get_local_size(0);
    size_t global_dim_size = get_global_size(0);
    size_t idx = get_global_id(0);

    // TODO: Make sure we don't hit overflow issues with swing/traction summation ratio.
    float minx, maxx, miny, maxy;
    float swing, traction;
    float val;
    int inc = global_dim_size;

    debugonce("bound box\n");


    // TODO: Make these kernel parameters, don't rely on macro
    __local float sminx[THREADS_BOUND], smaxx[THREADS_BOUND], sminy[THREADS_BOUND], smaxy[THREADS_BOUND];
    __local float local_swings[THREADS_BOUND], local_tractions[THREADS_BOUND];
    minx = maxx = x_cords[0];
    miny = maxy = y_cords[0];
    swing = 0.0f;
    traction = 0.0f;

    // For every body s.t. body_id % global_size == idx,
    // compute the min and max.
    for (int j = idx; j < num_bodies; j += inc) {
        val = x_cords[j];
        minx = min(val, minx);
        maxx = max(val, maxx);
        val = y_cords[j];
        miny = min(val, miny);
        maxy = max(val, maxy);
        swing += swings[j];
        traction += tractions[j];
    }

    // Standard reduction in shared memory to compute max/min for our
    // work group.
    sminx[tid] = minx;
    smaxx[tid] = maxx;
    sminy[tid] = miny;
    smaxy[tid] = maxy;
    local_swings[tid] = swing;
    local_tractions[tid] = traction;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int step = (dim / 2); step > 0; step = step / 2) {
        if (tid < step) {
            sminx[tid] = minx = min(sminx[tid] , sminx[tid + step]);
            smaxx[tid] = maxx = max(smaxx[tid], smaxx[tid + step]);
            sminy[tid] = miny = min(sminy[tid] , sminy[tid + step]);
            smaxy[tid] = maxy = max(smaxy[tid], smaxy[tid + step]);
            local_swings[tid] = swing = local_swings[tid] + local_swings[tid + step];
            local_tractions[tid] = traction = local_tractions[tid] + local_tractions[tid + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Only one thread needs to write value to the global buffer
    inc = (global_dim_size / dim) - 1;
    if (tid == 0) {
        global_x_mins[gid] = minx;
        global_x_maxs[gid] = maxx;
        global_y_mins[gid] = miny;
        global_y_maxs[gid] = maxy;

        // We use swings/tractions as buffer here, even though it's waaay larger than
        // necessary for this purpose.
        global_swings[gid] = swing;
        global_tractions[gid] = traction;

        inc = (global_dim_size / dim) - 1;
        if (inc == atomic_inc(blocked)) {

            // If we're in this block, it means we're the last work group
            // to execute. Find min/max among the global buffer.
            for(int j = 0; j <= inc; j++) {
                minx = min(minx, global_x_mins[j]);
                maxx = max(maxx, global_x_maxs[j]);
                miny = min(miny, global_y_mins[j]);
                maxy = max(maxy, global_y_maxs[j]);
                swing = swing + global_swings[j];
                traction = traction + global_tractions[j];
            }

            // Compute global speed
            if (step_number > 1) {
                // Use (exponential) rolling average of global speed to help smooth movement.
                /**globalSpeed = (0.1 * pow(10, tau) * (traction / swing)) + (0.9 * (*globalSpeed));*/
                // Don't use rolling average
                *globalSpeed = pow(10, tau) * (traction / swing);
            } else {
                *globalSpeed = 1.00f;
            }


            /*if (step_number == 100) {*/
                /**globalSpeed = 1.0f;*/
            /*}*/

            // Compute the radius
            val = max(maxx - minx, maxy - miny);
            *radiusd = (float) (val* 0.5f);

            // Create the root node at index num_nodes.
            // Because memory is laid out as bodies then tree-nodes,
            // k will be the first tree-node. We set its position to
            // the center of the bounding box, and initialize values.
            int k = num_nodes;
            *bottomd = k;
            mass[k] = -1.0f;
            start[k] = 0;
            x_cords[num_nodes] = (minx + maxx) * 0.5f;
            y_cords[num_nodes] = (miny + maxy) * 0.5f;

            // Set children to -1 ('null pointer')
            k *= 4;
            for (int i = 0; i < 4; i++) children[k + i] = NULLPOINTER;
            (*stepd)++;
        }
    }
}
