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
#define THREADS1 256    /* must be a power of 2 */ // Used for setting local buffer sizes.
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

// In theory this should be set dynamically, or the code should be rewritten to be
// warp agnostic (as is proper in OpenCL)
// Should be gotten by CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE parameter in the clGetKernelWorkGroupInfo().
// Pretty sure most modern NVidia have warp of 32, and AMD 'wavefront' of 64
// Correctness is guaranteed if WARPSIZE is less than or equal to actual warp size.
//#define WARPSIZE 32
#define MAXDEPTH 32
#define COMPUTE_SUMS_ITERATION_LIMIT 5000

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
    debugonce("to barnes layout\n");
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
    __local float sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1], smaxy[THREADS1];
    __local float local_swings[THREADS1], local_tractions[THREADS1];
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
        swings[gid] = swing;
        tractions[gid] = traction;

        inc = (global_dim_size / dim) - 1;
        if (inc == atomic_inc(blocked)) {

            // If we're in this block, it means we're the last work group
            // to execute. Find min/max among the global buffer.
            for(int j = 0; j <= inc; j++) {
                minx = min(minx, global_x_mins[j]);
                maxx = max(maxx, global_x_maxs[j]);
                miny = min(miny, global_y_mins[j]);
                maxy = max(maxy, global_y_maxs[j]);
                swing = swing + swings[j];
                traction = traction + tractions[j];
            }

            // Compute global speed
            if (step_number > 1) {
                *globalSpeed = min(tau * (traction / swing), *globalSpeed * 2);
            } else {
                *globalSpeed = 1.0f;
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

    debugonce("build tree\n");


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
                            // printf("BUILD TREE PROBLEM\n");
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
    volatile int missing_children[THREADS1 * 4];
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

        // TODO: Find a clean way to avoid this entirely.
        // It's likely impossible in OpenCL 1.2 because of lack of
        // global memory sync.
        if (iterations++ > COMPUTE_SUMS_ITERATION_LIMIT) {
            // printf("Compute sums iterations exceeded limit.\n");
            break;
        }

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

    debugonce("sort\n");

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
        float tau
){

    debugonce("calculate forces\n");

    const int idx = get_global_id(0);
    const int local_size = get_local_size(0);
    const int global_size = get_global_size(0);
    const int local_id = get_local_id(0);
    int k, index, i;
    float force;

    //float forceX, forceY;
    float2 forceVector;
    float2 distVector;



    float px, py, ax, ay, dx, dy, temp;
    int warp_id, starting_warp_thread_id, shared_mem_offset, difference, depth, child;

    // THREADS1/WARPSIZE is number of warps
    __local volatile int child_index[MAXDEPTH * THREADS1/WARPSIZE], parent_index[MAXDEPTH * THREADS1/WARPSIZE];
    __local volatile float dq[MAXDEPTH * THREADS1/WARPSIZE];

    __local volatile int shared_step, shared_maxdepth;
    __local int votingBuffer[THREADS1];

    if (local_id == 0) {
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
                            forceVector += normalize(distVector) * repulsionForce(distVector, 1.0,
                                            mass[child], scalingRatio, IS_PREVENT_OVERLAP(flags));
                                // forceVector += (float2) (0.000001f, 0.000001f);

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
            const float gForce = gravityForce(gravity, mass[index], centerVec, IS_STRONG_GRAVITY(flags));
            pointForces[index] = normalize(centerVec) * gForce + forceVector * mass[index];

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

    debugonce("from barnes layout\n");

    size_t gid = get_global_id(0);
    size_t global_size = get_global_size(0);
    for (int i = gid; i < numPoints; i += global_size) {
        outputPositions[i].x = x_cords[i];
        outputPositions[i].y = y_cords[i];
    }
}

