#include "common.h"
#include "layouts/forceAtlas2/barnesHut/barnesHutCommon.h"

__kernel void compute_sums(
        //graph params
        float scalingRatio, float gravity, unsigned int edgeWeightInfluence, unsigned int flags,
        __global volatile float *x_cords,
        __global float *y_cords,
        __global float* accx,
        __global float* accy,
        __global volatile int* children,
        __global float* mass,
        __global int* start,
        __global int* sort,
        __global float* global_x_mins,
        __global float* global_x_maxs,
        __global float* global_y_mins,
        __global float* global_y_maxs,
        __global float* swings,
        __global float* tractions,
        __global int* count,
        __global volatile int* blocked,
        __global volatile int* step,
        __global volatile int* bottom,
        __global volatile int* maxdepth,
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

    debugonce("compute sums\n");

    int i, j, k, inc, num_children_missing, cnt, bottom_value, child, local_size;
    float m, cm, px, py;

    // TODO: Should this be THREADS3 * 4 like in CUDA?
    volatile int missing_children[THREADS_SUMS * 4];
    // TODO cache kernel information

    bottom_value = *bottom;
    inc = get_global_size(0);
    local_size = get_local_size(0);

    // Align work to WARP SIZE
    // k is our iteration variable
    k = (bottom_value & (-WARPSIZE)) + get_global_id(0);
    if (k < bottom_value) k += inc;

    num_children_missing = 0;


    int iterations = 0;
    while (k <= num_nodes) {


        if (num_children_missing == 0) { // Must be new cell
            // Initialize
            cm = 0.0f;
            px = 0.0f;
            py = 0.0f;
            cnt = 0;
            j = 0;
            for (i = 0; i < 4; i++) {

                child = children[k*4+i];
                if (child > NULLPOINTER) {
                    if (i != j) {
                        // Moving children to front. Apparently needed later
                        // TODO figure out why this is
                        children[k*4+i] = -1;
                        children[k*4+j] = child;
                    }
                    // TODO: Make sure threads value is correct.

                    missing_children[num_children_missing*local_size+get_local_id(0)] = child;

                    m = mass[child];
                    num_children_missing++;
                    if (m >= 0.0f) {
                        // Child has already been touched
                        num_children_missing--;
                        if (child >= num_bodies) { // Count the bodies. TODO Why?
                            // TODO: Where is this initialized. Is it initialized to anything before?
                            cnt += count[child] - 1;
                        }

                        // Sum mass and position contributions
                        cm += m;
                        px += x_cords[child] * m;
                        py += y_cords[child] * m;
                    }
                    j++;
                }
            }
            cnt += j;
        }

        if (num_children_missing != 0) {
            do {
                // poll for missing child

                child = missing_children[(num_children_missing - 1)*local_size+get_local_id(0)];
                m = mass[child];
                if (m >= 0.0f) {
                    // Child has been touched
                    num_children_missing--;
                    if (child >= num_bodies) { // Count the bodies. TODO Why?
                        cnt += count[child] - 1;
                    }
                    // Sum mass and positions
                    cm += m;
                    px += x_cords[child] * m;
                    py += y_cords[child] * m;
                }
            } while ((m >= 0.0f) && (num_children_missing != 0));
                // Repeat until we are done or child is not ready TODO question: is this for thread divergence?
        }

        if (num_children_missing == 0) {
            //We're done! finish the sum

            count[k] = cnt;
            // Guard for case where children have no mass
            m = (cm == 0.0f) ? 0.0f : 1.0f / cm;
            x_cords[k] = px * m;
            y_cords[k] = py * m;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            mass[k] = cm;
            k += inc;
        }
        // make sure the change to `mass` is visable before the next iteration
        mem_fence(CLK_GLOBAL_MEM_FENCE);
    }
}
