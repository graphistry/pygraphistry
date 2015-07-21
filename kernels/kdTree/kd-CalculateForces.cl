#include "common.h"
#include "gsCommon.cl"
#include "barnesHut/barnesHutCommon.h"

#define HALF_WARP (WARPSIZE / 2)

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

#ifdef INLINEPTX
#define warpCellVote(buffer, distSquared, dq, warpId) ptx_thread_vote(distSquared, dq)
#else
#define warpCellVote(buffer, distSquared, dq, warpId) portable_all(buffer, distSquared, dq, warpId)
#endif

#ifdef INLINEPTX
inline uint ptx_thread_vote(float rSq, float rCritSq) {
    uint result = 0;
    asm("{\n\t"
         ".reg .pred cond, out;\n\t"
         "setp.ge.f32 cond, %1, %2;\n\t"
         "vote.all.pred out, cond;\n\t"
         "selp.u32 %0, 1, 0, out;\n\t"
         "}\n\t"
         : "=r"(result)
         : "f"(rSq), "f"(rCritSq));

    return result;
}
#endif

inline int portable_all(__local int volatile * buffer, const float distSquared, const float dq, const int warpId) {
    int cond = (distSquared >= dq);
    if (cond) {
        buffer[warpId] = 1;
    } else {
        buffer[warpId] = 0;
    }
    return buffer[warpId];
}

inline int reduction_thread_vote(__local int* const buffer, const float distSquared, const float dq, const int offset, const int diff) {
    // Relies on the fact that the wavefront/warp (not whole workgroup)
    // is executing in lockstep to avoid a barrier
    // Also, in C (and openCL) a conditional is an int value of 0 or 1
    int cond = (distSquared >= dq);

    int myoffset = offset + diff;
    buffer[myoffset] = cond;
    __local int* myBuffer = buffer + myoffset;


    // Runs in log_2(WARPSIZE) steps.
    for (int step = HALF_WARP; step > 0; step = step / 2) {
        if (diff < step) {
            myBuffer[0] = myBuffer[0] + myBuffer[step];
        }
    }

    return (buffer[offset] == WARPSIZE);
}

__kernel void calculate_forces(
        __global float *x_cords,
        __global float *y_cords,
        __global float* edgeDirectionX,
        __global float* edgeDirectionY,
        __global float* edgeLengths,
        __global int* children,
        __global int* sort,
        __global int* blocked,
        __global int* maxdepth,
        __global float* radiusd,
        unsigned int step_number,
        const int num_bodies,
        const int num_nodes,
        __global float2* pointForces,
        float charge,
        const uint midpoint_stride,
        const uint midpoints_per_edge
){

    debugonce("calculate forces\n");

    const int idx = get_global_id(0);
    const int local_size = get_local_size(0);
    const int global_size = get_global_size(0);
    const int local_id = get_local_id(0);
    /*const float alpha = max(0.1f * pown(0.99f, floor(convert_float(step_number) / (float) TILES_PER_ITERATION)), 0.005f);*/
    const float alpha = max(1.0f * pown(0.85f, step_number), 0.004f);
    if (idx == 0) {
        debug2("Alpha in forces: %f \n", alpha);
    }


    /*const float alpha = (float) TILES_PER_ITERATION;*/
    int k, index, i;
    float force;

    //float forceX, forceY;
    float2 forceVector;
    float2 distVector;

    // Edge compatability variables
    float edgeLengthOtherPoint;
    float edgeAngleCompat;
    float averageLength;
    /*float maxLength;*/
    /*float minLength;*/
    float edgeScaleCompat;
    float positCompat;
    float2 projectionVector;
    float midEdgeLength;
    float alignmentCompat;

    float px, py, ax, ay, dx, dy, temp;
    int warp_id, starting_warp_thread_id, shared_mem_offset, difference, depth, child;

    // THREADS1/WARPSIZE is number of warps
    __local volatile int child_index[MAXDEPTH * THREADS_FORCES/WARPSIZE], parent_index[MAXDEPTH * THREADS_FORCES/WARPSIZE];
    __local volatile float dq[MAXDEPTH * THREADS_FORCES/WARPSIZE];

    __local volatile int shared_maxdepth;
    __local volatile int votingBuffer[THREADS_FORCES/WARPSIZE];

    if (local_id == 0) {
        int itolsqd = 1.0f / (0.5f*0.5f);
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
            if ((index < 0) || (index > num_bodies)) {
                continue;
            }
            px = x_cords[index];
            py = y_cords[index];
            float2 normalizedPos = (float2) (px, py) / *radiusd;
            float edgeLength = edgeLengths[index];
            float edgeDirX = edgeDirectionX[index];
            float edgeDirY = edgeDirectionY[index];
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
                        float dist = length(distVector);
                        temp = dx*dx + (dy*dy + 0.00000000001);

                        // Edgebundling currently only uses the quadtree in order to test which points are close enough
                        // to compute forces on. It does not compute forces with any of the summarized nodes.
                        if ((child < num_bodies)) {
                            if (true || dist > FLT_EPSILON * 100.0f) {
                            const float2 otherPoint = (float2) (x_cords[child], y_cords[child]) / *radiusd;
                            edgeLengthOtherPoint = edgeLengths[child];
                            // TODO It would be interesting to to having edges of opposite
                            // direction repel instead of attract each other.
                            // TODO optimized registers.

                            edgeAngleCompat = pown((fmax((edgeDirectionX[child] * edgeDirX), 0) + fmax((edgeDirectionY[child] * edgeDirY), 0))/2, 3);
                            edgeAngleCompat = (edgeAngleCompat > 0.10f) ? edgeAngleCompat : 0.0f;
                            /*edgeAngleCompat = (fmax((edgeDirectionX[child] * edgeDirX), 0) + fmax((edgeDirectionY[child] * edgeDirY), 0))/ 2.0f;*/
                            averageLength = (edgeLength + edgeLengthOtherPoint) / 2.0f;
                            /*averageLength = (averageLength > 0.05f) ? averageLength : 0.0f;*/
                            edgeScaleCompat = 2.0f / ((max(edgeLength, edgeLengthOtherPoint) / averageLength) + (min(edgeLength, edgeLengthOtherPoint) / averageLength));
                            edgeScaleCompat = (edgeScaleCompat > 0.15f) ? edgeScaleCompat : 0.0f;
                            positCompat = averageLength / (averageLength + fast_length(distVector));
                            positCompat = (positCompat > 0.15f) ? positCompat : 0.0f;
                            projectionVector = dot(distVector, (float2) (edgeDirX, edgeDirY)) * (float2) (edgeDirX, edgeDirY);
                            projectionVector = (projectionVector > 0.15f) ? projectionVector : 0.0f;
                            midEdgeLength = edgeLength / midpoints_per_edge;
                            alignmentCompat = 1.0f / (1 + (fast_length(projectionVector)) / (midEdgeLength));
                            alignmentCompat = (alignmentCompat > 0.15f) ? alignmentCompat : 0.0f;
                            forceVector += /*pow((edgeLength / *radiusd), 1.0f) **/ edgeLength * alignmentCompat * edgeScaleCompat * edgeAngleCompat *  positCompat * (pointForce(normalizedPos, otherPoint, charge * alpha) * -1.0f);
                            }
                      // If all threads agree that cell is too far away, move on.
                        /*} else if (!(warpCellVote(votingBuffer, 100.0f * pow((dist - (edgeLength / 2.0f)), 2.0f), dq[depth], warp_id))) {*/
                        } else if (!(warpCellVote(votingBuffer, pow(dist , 2.0f), /*(edgeLength / *radiusd) **/ dq[depth],  warp_id))) {
                        /*} else if (!(warpCellVote(votingBuffer, (edgeLength / *radiusd) * pow(dist , 2.0f), dq[depth] / 4000.0f, warp_id))) {*/
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
            pointForces[(index * midpoints_per_edge) + midpoint_stride] = forceVector;
            /*pointForces[(index * midpoints_per_edge) + midpoint_stride] = (float2) (1.0f, 1.0f);*/
            /*pointForces[(index * midpoints_per_edge) + midpoint_stride] = forceVector + (pow((edgeLength / (*radiusd / 4)), 1.5f) * -500.0f * normalize((float2) (edgeDirY, -edgeDirX)));*/
            debug6("Force in calculate midpoints x (%u) %.9g, y %.9g Result x %.9g y %.9g\n", (index * midpoints_per_edge) + midpoint_stride, forceVector.x, forceVector.y, result.x, result.y);

        }
    }
}
