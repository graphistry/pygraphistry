#include "common.h"

#define REPULSION_OVERLAP 0.00000001f
#define EPSILON 1.0f
#define IS_PREVENT_OVERLAP(flags) (flags & 1)
#define IS_STRONG_GRAVITY(flags) (flags & 2)
#define IS_DISSUADE_HUBS(flags) (flags & 4)
#define IS_LIN_LOG(flags) (flags & 8)

#define DEFAULT_NODE_SIZE 0.000001f

// The length of the 'randValues' array
#define RAND_LENGTH 73

// BARNES HUT defintions.
// TODO We don't need all these
#define THREADS1 256    /* must be a power of 2 */
#define THREADS2 1024
#define THREADS3 1024
#define THREADS4 256
#define THREADS5 256
#define THREADS6 512

// block count = factor * #SMs
#define FACTOR1 3
#define FACTOR2 1
#define FACTOR3 1  /* must all be resident at the same time */
#define FACTOR4 1  /* must all be resident at the same time */
#define FACTOR5 5
#define FACTOR6 3

#define WARPSIZE 16
#define MAXDEPTH 32

// TODO: I've replaced comparisons >= 0 with > NULLPOINTER for readability.
// We should benchmark to make sure that doesn't impact perf.
#define TREELOCK -2
#define NULLPOINTER -1


//============================= BARNES HUT

// Computes BarnesHut specific data.
__kernel void to_barnes_layout(
        //GRAPH_PARAMS
        float scalingRatio, float gravity, unsigned int edgeWeightInfluence, unsigned int flags,
        // number of points
        unsigned int numPoints,
        const __global float2* inputPositions,
        __global float *x_cords,
        __global float *y_cords,
        __global float* mass,
        __global volatile int* blocked,
        __global volatile int* maxdepthd,
        const __global uint* pointDegrees,
        const uint step_number
){

    size_t gid = get_global_id(0);
    size_t global_size = get_global_size(0);

    for (int i = gid; i < numPoints; i += global_size) {
        x_cords[i] = inputPositions[i].x;
        y_cords[i] = inputPositions[i].y;
        mass[i] = (float) pointDegrees[i];
    }
    if (gid == 0) {
        *maxdepthd = -1;
        *blocked = 0;
    }
}


/*__attribute__ ((reqd_work_group_size(THREADS1, 1, 1)))*/
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
        __global int* count,
        __global volatile int* blocked,
        __global volatile int* stepd,
        __global volatile int* bottomd,
        __global volatile int* maxdepthd,
        __global volatile float* radiusd,
        unsigned int step_number,
        float width,
        float height,
        const int num_bodies,
        const int num_nodes,
        __global float2* pointForces
){

    size_t tid = get_local_id(0);
    size_t gid = get_group_id(0);
    size_t dim = get_local_size(0);
    size_t global_dim_size = get_global_size(0);
    size_t idx = get_global_id(0);

    float minx, maxx, miny, maxy;
    float val;
    int inc = global_dim_size;

    // TODO: Make these kernel parameters, don't rely on macro
    __local float sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1], smaxy[THREADS1];
    minx = maxx = x_cords[0];
    miny = maxy = y_cords[0];

    // For every body s.t. body_id % global_size == idx,
    // compute the min and max.
    for (int j = idx; j < num_bodies; j += inc) {
        val = x_cords[j];
        minx = min(val, minx);
        maxx = max(val, maxx);
        val = y_cords[j];
        miny = min(val, miny);
        maxy = max(val, maxy);
    }

    // Standard reduction in shared memory to compute max/min for our
    // work group.
    sminx[tid] = minx;
    smaxx[tid] = maxx;
    sminy[tid] = miny;
    smaxy[tid] = maxy;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int step = (dim / 2); step > 0; step = step / 2) {
        if (tid < step) {
            sminx[tid] = minx = min(sminx[tid] , sminx[tid + step]);
            smaxx[tid] = maxx = max(smaxx[tid], smaxx[tid + step]);
            sminy[tid] = miny = min(sminy[tid] , sminy[tid + step]);
            smaxy[tid] = maxy = max(smaxy[tid], smaxy[tid + step]);
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

        inc = (global_dim_size / dim) - 1;
        if (inc == atomic_inc(blocked)) {

            // If we're in this block, it means we're the last work group
            // to execute. Find min/max among the global buffer.
            for(int j = 0; j <= inc; j++) {
                minx = min(minx, global_x_mins[j]);
                maxx = max(maxx, global_x_maxs[j]);
                miny = min(miny, global_y_mins[j]);
                maxy = max(maxy, global_y_maxs[j]);
            }

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


__kernel void build_tree(
        //graph params
        float scalingRatio, float gravity, unsigned int edgeWeightInfluence, unsigned int flags,
        __global volatile float *x_cords,
        __global float *y_cords,
        __global float* accx,
        __global float* accy,
        __global volatile int* child,
        __global float* mass,
        __global int* start,
        __global int* sort,
        __global float* global_x_mins,
        __global float* global_x_maxs,
        __global float* global_y_mins,
        __global float* global_y_maxs,
        __global int* count,
        __global volatile int* blocked,
        __global volatile int* step,
        __global volatile int* bottom,
        __global volatile int* maxdepth,
        __global volatile float* radiusd,
        unsigned int step_number,
        float width,
        float height,
        const int num_bodies,
        const int num_nodes,
        __global float2* pointForces
){

    int inc =  get_global_size(0);
    int i = get_global_id(0);

    float radius = *radiusd;
    float rootx = x_cords[num_nodes];
    float rooty = y_cords[num_nodes];
    int localmaxdepth = 1;
    int skip = 1;

    float r;
    float x, y;
    int j;
    float px, py; // x and y of particle we're looking at
    int ch, n, cell, locked, patch;
    int depth;

    // Iterate through all bodies that were assigned to this thread
    // TODO: Make sure num_bodies is initialized to a proper value DYNAMICALLY
    while (i < num_bodies) {

        // If it's a new body, skip will be true, so we start at the root
        if (skip != 0) {
            skip = 0;
            px = x_cords[i];
            py = y_cords[i];
            n = num_nodes;
            depth = 1;
            r = radius;

            // j lets us know which of the 4 children to follow.
            j = 0;
            if (rootx < px) j = 1;
            if (rooty < py) j += 2;
        }

        // Walk down the path to a leaf node
        ch = child[n*4 + j];
        // We compare against num_bodies because it enforces two requirements.
        // If it's a body, then we know we've gone past the legal space for
        // cells. If it's a 'null' child, then it'll be initialized to -1, which
        // is also < num_bodies.
        while (ch >= num_bodies) {
            n = ch;
            depth++;
            r *= 0.5f;
            j = 0;
            // determine which child to follow
            if (x_cords[n] < px) j = 1;
            if (y_cords[n] < py) j += 2;
            ch = child[n*4+j];
        }

        // Skip if the child is currently locked.
        if (ch != TREELOCK) {
            locked = n*4+j;

            // Attempt to lock the child
            if (ch == atomic_cmpxchg(&child[locked], ch, TREELOCK)) {
                // TODO: Determine if we need this fence
                //mem_fence(CLK_GLOBAL_MEM_FENCE);

                // If the child was null, just insert the body.
                if (ch == NULLPOINTER) {
                    child[locked] = i;
                } else {
                    patch = NULLPOINTER;
                    // create new cell(s) and insert the old and new body

                    // TODO: Do we still need test?
                    int test = 1000000;
                    do {

                        // We allocate from right to left, so we use an atomic_dec
                        depth++;
                        cell = atomic_dec(bottom) - 1;

                        // Error case
                        if (cell <= num_bodies) {
                            // TODO (paden) add error message
                            *bottom = num_nodes;
                            return;
                        }

                        patch = max(patch, cell);

                        x = (j & 1) * r;
                        y = ((j >> 1) & 1) * r;
                        r *= 0.5f;

                        mass[cell] = -1.0f;
                        start[cell] = NULLPOINTER;
                        x = x_cords[cell] = x_cords[n] - r + x;
                        y = y_cords[cell] = y_cords[n] - r + y;

                        // TODO: Unroll
                        // Initialize new children to null.
                        for (int k = 0; k < 4; k++) child[cell*4+k] = NULLPOINTER;

                        // Make it point to the cell if cell was greater than patch.
                        // This means that this is the first time this cell is accessed,
                        // and thus that it needs to be pointed to.
                        if (patch != cell) {
                            child[n*4+j] = cell;
                        }

                        // Place already existing body from before into the correct
                        // child node.
                        j = 0;
                        if (x < x_cords[ch]) j = 1;
                        if (y < y_cords[ch]) j += 2;
                        child[cell*4+j] = ch;

                        // If they have the exact same location, shift one slightly.
                        // TODO: Do we need this? Not in original CUDA impl.
                        if (x_cords[ch] == px && y_cords[ch] == py) {
                            x_cords[ch] += 0.0000001;
                            y_cords[ch] += 0.0000001;
                        }

                        n = cell;
                        j = 0;
                        if (x < px) j = 1;
                        if (y < py) j += 2;

                        // Updated ch to the child when our px/py body will go.
                        // If it's -1 (null) we exit out of this loop.
                        ch = child[n*4+j];

                        // TODO: Do we still need this?
                        test--;

                    } while (test > 0 && ch > NULLPOINTER);

                    // Place our body and expose to other threads.
                    child[n*4+j] = i;
                    mem_fence(CLK_GLOBAL_MEM_FENCE); // push out our subtree to other threads.
                    child[locked] = patch;
                }

                localmaxdepth = max(depth, localmaxdepth);
                i += inc;  // move on to next body
                skip = 1;
            }
        }
        // TODO: Uncomment this throttle after testing:
        // barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Record our maximum depth globally.
    atomic_max(maxdepth, localmaxdepth);
}


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
        __global int* count,
        __global volatile int* blocked,
        __global volatile int* step,
        __global volatile int* bottom,
        __global volatile int* maxdepth,
        __global volatile float* radiusd,
        unsigned int step_number,
        float width,
        float height,
        const int num_bodies,
        const int num_nodes,
        __global float2* pointForces
){

    int i, j, k, inc, num_children_missing, cnt, bottom_value, child;
    float m, cm, px, py;

    // TODO: Should this be THREADS3 * 4 like in CUDA?
    volatile int missing_children[THREADS1 * 4];
    // TODO cache kernel information

    bottom_value = *bottom;
    inc = get_global_size(0);

    // Align work to WARP SIZE
    // k is our iteration variable
    k = (bottom_value & (-WARPSIZE)) + get_global_id(0);
    if (k < bottom_value) k += inc;

    num_children_missing = 0;

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
                    missing_children[num_children_missing*THREADS1+get_local_id(0)] = child;
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
                child = missing_children[(num_children_missing - 1)*THREADS1+get_local_id(0)];
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
            m = 1.0f / cm;
            x_cords[k] = px * m;
            y_cords[k] = py * m;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            mass[k] = cm;
            k += inc;
        }
    }
}


// Sort bodies in in-order traversal order
__kernel void sort(
        //graph params
        float scalingRatio, float gravity, unsigned int edgeWeightInfluence, unsigned int flags,
        __global volatile float *x_cords,
        __global float *y_cords,
        __global float* accx,
        __global float* accy,
        __global volatile int* children,
        __global float* mass,
        __global volatile int* start,
        __global int* sort,
        __global float* global_x_mins,
        __global float* global_x_maxs,
        __global float* global_y_mins,
        __global float* global_y_maxs,
        __global int* count,
        __global volatile int* blocked,
        __global volatile int* step,
        __global volatile int* bottom,
        __global volatile int* maxdepth,
        __global volatile float* radiusd,
        unsigned int step_number,
        float width,
        float height,
        const int num_bodies,
        const int num_nodes,
        __global float2* pointForces
){

    int i, k, child, decrement, start_index, bottom_node;

    bottom_node = *bottom;
    decrement = get_global_size(0);
    k = num_nodes + 1 - decrement + get_global_id(0);

    while (k >= bottom_node) {
        start_index = start[k];
        if (start_index >= 0) {
            for (i = 0; i < 4; i++) {
                child = children[k*4+i];
                if (child >= num_bodies) {
                    // Child must be a cell
                    start[child] = start_index; // Set start ID of child
                    start_index += count[child]; // Add number of bodies in subtree
                } else if (child >= 0) {
                    // Child must be a body
                    sort[start_index] = child; // Record the body in 'sorted' array
                    start_index++;
                }
            }
            k -= decrement; // Go to next cell
        }
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        //barrier(CLK_GLOBAL_MEM_FENCE); //TODO how to add throttle?
    }
}


inline int thread_vote(__local int* allBlock, int warpId, int cond)
{
     /*printf("in thread vote\n");*/
     /*Relies on underlying wavefronts (not whole workgroup)*/
         /*executing in lockstep to not require barrier */
    int old = allBlock[warpId];

    // Increment if true, or leave unchanged
    (void) atomic_add(&allBlock[warpId], cond);

    int ret = (allBlock[warpId] == WARPSIZE);
    /*printf("Return : %d , num: %d \n", ret, allBlock[warpId]);*/
    /*printf("allBlock[warp]: %d warp %d \n", allBlock[warpId], warpId);*/
    allBlock[warpId] = old;

    return ret;
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
        float scalingRatio, float gravity, unsigned int edgeWeightInfluence, unsigned int flags,
        __global volatile float *x_cords,
        __global float *y_cords,
        __global float *accx,
        __global float * accy,
        __global volatile int* children,
        __global float* mass,
        __global int* start,
        __global int* sort,
        __global float* global_x_mins,
        __global float* global_x_maxs,
        __global float* global_y_mins,
        __global float* global_y_maxs,
        __global int* count,
        __global volatile int* blocked,
        __global volatile int* step,
        __global volatile int* bottom,
        __global volatile int* maxdepth,
        __global volatile float* radiusd,
        unsigned int step_number,
        float width,
        float height,
        const int num_bodies,
        const int num_nodes,
        __global float2* pointForces
){

    int idx = get_global_id(0);
    int k, index, i;
    float force;
    float2 forceVector;
    int warp_id, starting_warp_thread_id, shared_mem_offset, difference, depth, child;
    __local volatile int child_index[MAXDEPTH * THREADS1/WARPSIZE], parent_index[MAXDEPTH * THREADS1/WARPSIZE];
     __local volatile int allBlock[THREADS1 / WARPSIZE];
    __local volatile float dq[MAXDEPTH * THREADS1/WARPSIZE];
    __local volatile int shared_step, shared_maxdepth;
    __local volatile int allBlocks[THREADS1/WARPSIZE];
    const unsigned int tileSize = (unsigned int) get_local_size(0);
    const unsigned int numTiles = (unsigned int) get_num_groups(0);
    unsigned int modulus = numTiles / TILES_PER_ITERATION;
    unsigned int startTile = (step_number % modulus) * (tileSize * TILES_PER_ITERATION);
    unsigned int endTile = (startTile + (tileSize * TILES_PER_ITERATION)) > num_bodies ? num_bodies : startTile + (tileSize * TILES_PER_ITERATION);
    unsigned int number_elements = (endTile > num_nodes) ? endTile - num_nodes : tileSize;
    float px, py, ax, ay, dx, dy, temp;
    int global_size = get_global_size(0);
    float2 distVector;
    if (get_local_id(0) == 0) {
        /*printf("Number of groups %u\n", numTiles);*/
        /*printf("startTile %u \n", startTile);*/
        /*printf("modulus %u \n", modulus);*/
        /*printf("endTile %u \n", endTile);*/
        /*printf("number_of %u \n", number_elements);*/
        /*printf("number of tiles %u \n", numTiles);*/
        /*printf("step number %d \n", step_number);*/
        int itolsqd = 1.0f / (0.5f*0.5f);
        shared_step = *step;
        shared_maxdepth = *maxdepth;
        temp = *radiusd;
        dq[0] = temp * temp * itolsqd;
        for (i = 1; i < shared_maxdepth; i++) {
            dq[i] = dq[i - 1] * 0.25f;
        }

        if (shared_maxdepth > MAXDEPTH) {
            return;
            //temp =    1/0;
        }
        for (i = 0; i < THREADS1/WARPSIZE; i++) {
            allBlocks[i] = 0;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    if (shared_maxdepth <= MAXDEPTH) {
        // Warp and memory ids
        warp_id = get_local_id(0) / WARPSIZE;
        starting_warp_thread_id = warp_id * WARPSIZE;
        shared_mem_offset = warp_id * MAXDEPTH;
        difference = get_local_id(0) - starting_warp_thread_id;
        if (difference < MAXDEPTH) {
            dq[difference + shared_mem_offset] = dq[difference];
        }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    /*bodies_in_tile = (stepNumber % modulus */
    for (k = idx; k < /*number_elements*/num_bodies; k+=global_size) {
        //atomic_add(&allBlock[warp_id], 1);
        index = sort[k];
        px = x_cords[index];
        py = y_cords[index];
        ax = 0.0f;
        ay = 0.0f;
        forceVector = (float2) (0.0f, 0.0f);
        depth = shared_mem_offset;
        if (starting_warp_thread_id == get_local_id(0)) {
            parent_index[shared_mem_offset] = num_nodes;
            child_index[shared_mem_offset] = 0;
        }
        mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
        while (depth >= shared_mem_offset) {
            // Stack has elements
            while(child_index[depth] < 4) {
                child = children[parent_index[depth]*4+child_index[depth]];
                if (get_local_id(0) == starting_warp_thread_id) {
                    child_index[depth]++;
                }
                mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
                if (child >= 0) {
                    /*dx = x_cords[child] - px;*/
                    /*dy = y_cords[child] - py;*/

                    dx = px - x_cords[child];
                    dy = py - y_cords[child];
                    distVector = (float2) (dx, dy);
                    temp = dx*dx + (dy*dy + 0.00000000001);
                    /*printf("temp %f, dq[depth] %f\n", temp, dq[depth]);*/
                    if ((child < num_bodies)    ||  thread_vote(allBlocks, warp_id, temp >= dq[depth]) )    {
                        force = mass[child] / temp;
                        ax += dx * force;
                        ay += dy * force;

                        // Adding all forces
                        forceVector += normalize(distVector) * repulsionForce(distVector, mass[index],
                                        mass[child], scalingRatio, IS_PREVENT_OVERLAP(flags));


                    } else {
                        depth++;
                        if (starting_warp_thread_id == get_local_id(0)) {
                            parent_index[depth] = child;
                            child_index[depth] = 0;
                        }
                        mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
                    }
                } else {
                    depth = max(shared_mem_offset, depth - 1);
                }
            }
            depth--;
        }
        accx[index] = ax;
        accy[index] = ay;

        // Assigning force for force atlas.
        float2 n1Pos = (float2) (px, py);
        const float2 dimensions = (float2) (width, height);
        const float2 centerVec = (dimensions / 2.0f) - n1Pos;
        const float gForce = gravityForce(gravity, mass[index], centerVec, IS_STRONG_GRAVITY(flags));
        pointForces[index] = normalize(centerVec) * gForce + forceVector;

        }
    }
}


__kernel void move_bodies(
        float scalingRatio, float gravity, unsigned int edgeWeightInfluence, unsigned int flags,
        __global volatile float *x_cords,
        __global float *y_cords,
        __global float *accx,
        __global float * accy,
        __global volatile int* children,
        __global float* mass,
        __global int* start,
        __global int* sort,
        __global float* global_x_mins,
        __global float* global_x_maxs,
        __global float* global_y_mins,
        __global float* global_y_maxs,
        __global int* count,
        __global volatile int* blocked,
        __global volatile int* step,
        __global volatile int* bottom,
        __global volatile int* maxdepth,
        __global volatile float* radiusd,
        unsigned int step_number,
        float width,
        float height,
        const int num_bodies,
        const int num_nodes,
        __global float2* pointForces
){

    /*const float dtime = 0.025f;*/
    /*const float dthf = dtime * 0.5f;*/
    float velx, vely;

    /*printf("Gravity: %f\n", gravity);*/
    int inc = get_global_size(0);
    for (int i = get_global_id(0); i < num_bodies; i+= inc) {
        /*velx = accx[i] * dthf;*/
        /*vely = accy[i] * dthf;*/
        float center_distance_x = width/2 - x_cords[i];
        float center_distance_y = height/2 - y_cords[i];
        float gravity_force = gravity / sqrt(center_distance_x*center_distance_x + center_distance_y * center_distance_y);
        velx = (accx[i] * 0.00001f) + (0.01f * gravity_force * center_distance_x);
        vely = (accy[i] * 0.00001f) + (0.01f * gravity_force * center_distance_y);

        x_cords[i] += velx;
        y_cords[i] += vely;
    }
}


__kernel void from_barnes_layout(
        //GRAPH_PARAMS
        float scalingRatio, float gravity, unsigned int edgeWeightInfluence, unsigned int flags,
        // number of points
        unsigned int numPoints,
        const __global float2* outputPositions,
        __global float *x_cords,
        __global float *y_cords,
        __global float* mass,
        __global volatile int* blocked,
        __global volatile int* maxdepthd,
        unsigned int step_number
){

    size_t gid = get_global_id(0);
    size_t global_size = get_global_size(0);
    for (int i = gid; i < numPoints; i += global_size) {
        outputPositions[i].x = x_cords[i];
        outputPositions[i].y = y_cords[i];
    }
}

