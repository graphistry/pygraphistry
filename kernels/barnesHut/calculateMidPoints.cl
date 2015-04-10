#include "common.h"
#include "gsCommon.cl"
#include "barnesHut/barnesHutCommon.h"

inline int thread_vote(__local int* allBlock, int warpId, int cond)
{
     /*Relies on underlying wavefronts (not whole workgroup)*/
         /*executing in lockstep to not require barrier */
    int old = allBlock[warpId];
    //printf("Cond: %d\n", cond);

    // Increment if true, or leave unchanged
    (void) atomic_add(&allBlock[warpId], cond);

    int ret = (allBlock[warpId] == WARPSIZE);
    /*printf("Return : %d , num: %d \n", ret, allBlock[warpId]);*/
    /*printf("allBlock[warp]: %d warp %d \n", allBlock[warpId], warpId);*/
    allBlock[warpId] = old;

    return ret;
}

inline int reduction_thread_vote(__local int* buffer, int cond, int offset, int diff) {
    // Relies on the fact that the wavefront/warp (not whole workgroup)
    // is executing in lockstep to avoid a barrier
    // Also, in C (and openCL) a conditional is an int value of 0 or 1
    int myoffset = offset + diff;
    buffer[myoffset] = cond;

    // Runs in log_2(WARPSIZE) steps.
    for (int step = WARPSIZE / 2; step > 0; step = step / 2) {
        if (diff < step) {
            buffer[myoffset] = buffer[myoffset] + buffer[myoffset + step];
        }
    }
    return (buffer[offset] == WARPSIZE);
}



inline float repulsionForce(const float2 distVec, const uint n1Degree, const uint n2Degree,
                            const float scalingRatio, const bool preventOverlap) {
    const float dist = length(distVec);
    const int degreeProd = (n1Degree + 1) * (n2Degree + 1);
    float force;

    if (preventOverlap) {
        //FIXME include in prefetch etc, use actual sizes
        float n1Size = DEFAULT_NODE_SIZE;
        float n2Size = DEFAULT_NODE_SIZE;
        float distB2B = dist - n1Size - n2Size; //border-to-border

        force = distB2B > EPSILON  ? (scalingRatio * degreeProd / dist)
            : distB2B < -EPSILON ? (REPULSION_OVERLAP * degreeProd)
            : 0.0f;
    } else {
        force = scalingRatio * degreeProd / dist;
    }

#ifndef NOREPULSION
    return clamp(force, 0.0f, 1000000.0f);
#else
    return 0.0f;
#endif
}


inline float gravityForce(const float gravity, const uint n1Degree, const float2 centerVec,
                          const bool strong) {

    const float gForce = gravity *
            (n1Degree + 1.0f) *
            (strong ? length(centerVec) : 1.0f);
#ifndef NOGRAVITY
    return gForce;
#else
    return 0.0f;
#endif
}


__kernel void calculate_forces(
        //graph params
        const float scalingRatio, const float gravity, const unsigned int edgeWeightInfluence, const unsigned int flags,
        __global float *x_cords,
        __global float *y_cords,
        __global float *accx,
        __global float * accy,
        __global int* children,
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
        __global int* blocked,
        __global int* step,
        __global int* bottom,
        __global int* maxdepth,
        __global float* radiusd,
        __global volatile float* globalSpeed,
        unsigned int step_number,
        float width,
        float height,
        const int num_bodies,
        const int num_nodes,
        __global float2* pointForces,
        float tau,
        float charge,
        const uint midpoint_stride,
        const uint midpoints_per_edge
){

    /*if (gid == get_global_id(0)) {*/
        /*printf("Stride %d, midpoints_per_edge %d \n", midpoint_stride, midpoints_per_edge);*/
    /*}*/
    debugonce("calculate forces\n");

    const int idx = get_global_id(0);
    const int local_size = get_local_size(0);
    const int global_size = get_global_size(0);
    const int local_id = get_local_id(0);
    const float alpha = max(0.1f * pown(0.99f, floor(convert_float(step_number) / (float) TILES_PER_ITERATION)), 0.005f);

    /*const float alpha = (float) TILES_PER_ITERATION;*/
    int k, index, i;
    float force;

    //float forceX, forceY;
    float2 forceVector = (0.0f, 0.0f);
    float2 distVector = (0.0f, 0.0f);



    float px, py, ax, ay, dx, dy, temp;
    int warp_id, starting_warp_thread_id, shared_mem_offset, difference, depth, child;

    // THREADS1/WARPSIZE is number of warps
    __local volatile int child_index[MAXDEPTH * THREADS1/WARPSIZE], parent_index[MAXDEPTH * THREADS1/WARPSIZE];
    __local volatile float dq[MAXDEPTH * THREADS1/WARPSIZE];

    __local volatile int shared_step, shared_maxdepth;
    __local int votingBuffer[THREADS1];

    if (local_id == 0) {
      /*printf("num_nodes %d, num_bodies %d \n", num_nodes, num_bodies);*/
        int itolsqd = 1.0f / (0.5f*0.5f);
        shared_step = *step; // local
        shared_maxdepth = *maxdepth; // local
        temp = *radiusd; // local

        // dq is about 4 * radius^2 = 4*area.
        // each dq[i] is dq[i-1]/4 (because splitting into 4 at a new depth)
        // This is so at a given depth, we know a minimum distance before we can start
        // approximating with centers of mass.
        //
        // e.g., at depth 10 we have a distance of 4 * radius^2 * (0.25^10) before we can approximate
        dq[0] = temp * temp * itolsqd;
        for (i = 1; i < shared_maxdepth; i++) {
            dq[i] = dq[i - 1] * 0.25f;
        }
        if (shared_maxdepth > MAXDEPTH) {
            // TODO: Actual error handling here
            // printf("MAXDEPTH PROBLEM\n");

            // This is temporarily commented out to run on Intel Iris.
            // return;
        }
        //TODO: Do we haaave to do this?
        // for (i = 0; i < THREADS1/WARPSIZE; i++) {
        //     allBlocks[i] = 0;
        // }
    }

    //barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    barrier(CLK_LOCAL_MEM_FENCE); // Should only need local mem fence

    if (shared_maxdepth <= MAXDEPTH) {
        // Warp and memory ids
        warp_id = local_id / WARPSIZE;
        starting_warp_thread_id = warp_id * WARPSIZE;
        shared_mem_offset = warp_id * MAXDEPTH;
        difference = local_id - starting_warp_thread_id;

        if (difference < MAXDEPTH) {
            // This makes it so dq is laid out in memory like so:
            //
            // Idx:    0      1      2            WARPSIZE   WARPSIZE+1
            // Arr: [4*r^2, r^2, 0.25*r^2, ... ,   4*r^2,    r^2,
            //
            // This way each warp has access to its own copy of dq which was initialized earlier.

            dq[difference + shared_mem_offset] = dq[difference];
        }

        //barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
        barrier(CLK_LOCAL_MEM_FENCE); // Should only need local mem fence

        // Iterate through bodies this thread is responsible for.
        for (k = idx; k < num_bodies; k+=global_size) {
            index = sort[k];
            px = x_cords[index];
            py = y_cords[index];
            forceVector = (float2) (0.0f, 0.0f);
            depth = shared_mem_offset; // We use this to both keep track of depth and index into dq.

            // initialize iteration stack, i.e., push root node onto stack
            if (starting_warp_thread_id == local_id) {
                parent_index[shared_mem_offset] = num_nodes; // num_nodes is the first cell (root)
                child_index[shared_mem_offset] = 0;
            }

            // Make sure it's visible
            // mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            mem_fence(CLK_LOCAL_MEM_FENCE);

            while (depth >= shared_mem_offset) {
                // Because we haven't exited out of the depth=0 (shared_mem_offset), we still need to
                // look at some elements on this stack.
                while(child_index[depth] < 4) {
                    // Node on top of stack still has children
                    child = children[parent_index[depth]*4+child_index[depth]]; // load the child pointer
                    if (local_id == starting_warp_thread_id) {
                        // Only the first thread per warp does this, signifying we looked at one child
                        child_index[depth]++;
                    }

                    //mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
                    mem_fence(CLK_LOCAL_MEM_FENCE);
                    if (child > NULLPOINTER) {

                        // Compute distances
                        dx = px - x_cords[child];
                        dy = py - y_cords[child];
                        distVector = (float2) (dx, dy);
                        temp = dx*dx + (dy*dy + 0.00000000001);

                        if ((child < num_bodies) || reduction_thread_vote(votingBuffer, temp >= dq[depth], starting_warp_thread_id, difference)) {
                            // check if ALL threads agree that cell is far enough away (or is a body)

                            // TODO: Determine how often we diverge when a few threads see a body, and the
                            // rest fail because of insufficient distance.

                            // Adding all forces
                          float2 n1Pos = (float2) (px, py);
                          float2 distVector = (float2) (dx, dy);
                          float2 otherPoint = (float2) (x_cords[child], y_cords[child]);
                          /*float err = fast_distance(otherPoint, myPos);*/
                          if (fast_length(distVector) < FLT_EPSILON) {
                            forceVector += 0.00001f * pointForce(n1Pos, otherPoint, alpha* charge * mass[child]);
                          } else {
                            forceVector += 1.0f * (pointForce(n1Pos, otherPoint, charge * alpha * mass[child]) * -1.0f);
                          }
                        } else {
                            // Push this cell onto the stack.
                            depth++;
                            if (starting_warp_thread_id == local_id) {
                                parent_index[depth] = child;
                                child_index[depth] = 0;
                            }
                            //mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
                            mem_fence(CLK_LOCAL_MEM_FENCE); // Make sure it's visible to other threads
                        }

                    } else {
                        depth = max(shared_mem_offset, depth - 1); // Exit early because all other children would be 0
                    }
                }
                depth--; // Finished this level
            }

            // Assigning force for force atlas.
            float2 n1Pos = (float2) (px, py);
            const float2 dimensions = (float2) (width, height);
            const float2 centerVec = (dimensions / 2.0f) - n1Pos;
            const float gForce = gravityForce(gravity, /*mass[index]*/1.0f, centerVec, IS_STRONG_GRAVITY(flags));
            /*const float2 gForce2 = normalize(centerVec) * gForce * 0.000001f;*/
            /*forceVector += gForce2;*/
                            if (get_global_id(0) < 64) {
                              /*printf("Force x %f, Force y %f \n gForce x %f y %f \n", forceVector.x, forceVector.y, gForce2.x, gForce2.y);*/
                              /*printf("gForce x %.9g y %.9g x %.9g y %9g mass %f gravity %f\n", gForce2.x, gForce2.y, centerVec.x, centerVec.y, mass[index], gForce);*/
                            }
            pointForces[(index * midpoints_per_edge) + midpoint_stride] = n1Pos * forceVector;
            /*nextMidPoints[index] = n1Pos + 0.00001f * normalize(centerVec) * gForce + forceVector * mass[index];*/

        }
    }
}
