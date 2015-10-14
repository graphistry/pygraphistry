#include "common.h"
#include "layouts/forceAtlas2/barnesHut/barnesHutCommon.h"

#ifdef INLINEPTX
#define threadfenceWrapper() asm("{\n\t membar.gl;\n\t }\n\t")
#else
#define threadfenceWrapper() mem_fence(CLK_GLOBAL_MEM_FENCE)
#endif

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

        // Skip duplicate points or points below max depth
        if (depth >= MAXDEPTH || (fabs(px - x_cords[n]) <= FLT_EPSILON) && (fabs(py - y_cords[n]) <= FLT_EPSILON)) {
          i += inc;  // move on to next body
          skip = 1;
          continue;
        }


        mem_fence(CLK_GLOBAL_MEM_FENCE);
        // Skip if the child is currently locked.
        if (ch != TREELOCK) {
            locked = n*4+j;

            // Attempt to lock the child
            if (ch == atomic_cmpxchg(&child[locked], ch, TREELOCK)) {
                // TODO: Determine if we need this fence
                mem_fence(CLK_GLOBAL_MEM_FENCE);

                // If the child was null, just insert the body.
                if (ch == NULLPOINTER) {
                    child[locked] = i;
                } else {
                    patch = NULLPOINTER;
                    // create new cell(s) and insert the old and new body

                    do {

                        // We allocate from right to left, so we use an atomic_dec
                        depth++;
                        cell = atomic_dec(bottom) - 1;

                        // Error case
                        if (cell <= num_bodies) {
                            // TODO (paden) add error message
                             /*printf("BUILD TREE PROBLEM\n");*/
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


                        n = cell;
                        j = 0;
                        if (x < px) j = 1;
                        if (y < py) j += 2;

                        // Updated ch to the child when our px/py body will go.
                        // If it's -1 (null) we exit out of this loop.
                        ch = child[n*4+j];

                        // If child cannot position is perfectly equal to current node
                        // position. Just insert node arbitrarily. This should happen
                        // so rarely and at such a low depth, that the approximation
                        // should be tribial.
                        if (depth >= MAXDEPTH || ((fabs(px - x_cords[ch]) <= FLT_EPSILON) && (fabs(py - y_cords[ch]) <= FLT_EPSILON) && (ch != -1))) {
                          j = 0;
                          while ((ch = child[n*4 + j]) > NULLPOINTER && j < 3) j++;
                          // Even if child node has filled leaves, set ch to -1. This is a slightly
                          // larger approximation, but makes sure nothing breaks.
                          ch = -1;
                        }



                    } while (ch > NULLPOINTER);

                    // Place our body and expose to other threads.
                    child[n*4+j] = i;
                    threadfenceWrapper();
                    child[locked] = patch;
                }

                localmaxdepth = max(depth, localmaxdepth);
                i += inc;  // move on to next body
                skip = 1;
            }
        }
        // TODO: Uncomment this throttle after testing:
        // Technically not valid opencl, but valid CUDA/PTX.
        // This will work on an nvidia platform that allows for inline PTX
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        #ifdef INLINEPTX
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        #endif
    }
    // Record our maximum depth globally.
    atomic_max(maxdepth, localmaxdepth);
}
