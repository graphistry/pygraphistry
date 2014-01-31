/*
* Copyright (C) 2011 Samsung Electronics Corporation. All rights reserved.
* 
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided the following conditions
* are met:
* 
* 1.  Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
* 
* 2.  Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in the
*     documentation and/or other materials provided with the distribution.
* 
* THIS SOFTWARE IS PROVIDED BY SAMSUNG ELECTRONICS CORPORATION AND ITS
* CONTRIBUTORS "AS IS", AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING
* BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SAMSUNG
* ELECTRONICS CORPORATION OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
* INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
* DATA, OR PROFITS, OR BUSINESS INTERRUPTION), HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING
* NEGLIGENCE OR OTHERWISE ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
* EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


// For a single point in the graph, calculate the forces from each other point in the graph on it,
// and write out an updated position and velocity for it.
__kernel void nbody_kernel_GPU(
    __global float* curPos,     // Input position (read-only)
    __global float* curVel,     // Input velocities (read-only)
    int numBodies,              // Total number of points
    float deltaTime,            // Always 0.005 currently
    int epsSqr,                 // Always 50 currently
    __local float* localPos,    // Array at least big enough to hold one point from each workgroup
    __global float* nxtPos,     // Output positions (write-only)
    __global float* nxtVel)     // Output velocities (write-only)
{
    unsigned int tid = get_local_id(0);
    unsigned int gid = get_global_id(0);
    // Threads in this workgroup
    unsigned int localSize = get_local_size(0);

    // Number of tiles we need to iterate, i.e. the number of workgroups
    unsigned int numTiles = numBodies / localSize;

    // position of this work-item, i.e. the point we're updating
    float4 myPos = (float4) (curPos[4*gid + 0], curPos[4*gid + 1], curPos[4*gid + 2], curPos[4*gid + 3]);
    float4 acc = (float4) (0.0f, 0.0f, 0.0f, 0.0f);

    // For optimized memory access, we break up the set of points we're currently considering into
    // 'tiles'. The number of points in a tile is equal to the number of threads each workgroup has,
    // so we use all the threads in this workgroup to fetch a point from the tile into local memory,
    // sync, calculate the forces from all those points, sync, then fetch the next tile, and repeat.
    for(int i = 0; i < numTiles; ++i)
    {
        // The syncs in this loop mean that all threads in this workgroup are operating on the same
        // tile at the same time.
        
        // For the current tile, grab one point from it. Because there are an equal number of points
        // in a tile as there are threads in a workgroup (all workgroups have the same # of threads,
        // which are even divisible by the total # of threads/points,) when all threads are done,
        // all points from that workgroup have been fetched into local memory.
        int idx = i * localSize + tid;
        for(int k=0; k<4; k++)
        {
                localPos[4*tid+k] = curPos[4*idx+k];
        }
        // Synchronize to make sure data is available for processing
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // For all the points from the current tile, calculate acceleration effect due to each body
        // a[i->j] = m[j] * r[i->j] / (r^2 + epsSqr)^(3/2)
        for(int j = 0; j < localSize; ++j)
        {
            // Calculate acceleration caused by particle j on particle i
            float4 aLocalPos = (float4) (localPos[4*j + 0], localPos[4*j + 1], localPos[4*j + 2], localPos[4*j + 3]);
            float4 r = aLocalPos - myPos;
            float distSqr = r.x * r.x  +  r.y * r.y  +  r.z * r.z;
            float invDist = 1.0f / sqrt(distSqr + epsSqr);
            float invDistCube = invDist * invDist * invDist;
            float s = aLocalPos.w * invDistCube;
            // accumulate effect of all particles
            acc += s * r;
        }
        // Synchronize so that next tile can be loaded
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float4 oldVel = (float4) (curVel[4*gid + 0], curVel[4*gid + 1], curVel[4*gid + 2], curVel[4*gid + 3]);

    // updated position and velocity
    float4 newPos = myPos + (oldVel * deltaTime) + (acc * 0.5f * (deltaTime * deltaTime));
    newPos.w = myPos.w;
    float4 newVel = oldVel + (acc * deltaTime);

    // check boundry
    if(newPos.x > 1.0f || newPos.x < -1.0f || newPos.y > 1.0f || newPos.y < -1.0f || newPos.z > 1.0f || newPos.z < -1.0f) {
        float rand = (1.0f * gid) / numBodies;
        float r = 0.05f *  rand;
        float theta = rand;
        float phi = 2 * rand;
        newPos.x = r * sinpi(theta) * cospi(phi);
        newPos.y = r * sinpi(theta) * sinpi(phi);
        newPos.z = r * cospi(theta);
        newVel.x = 0.0f;
        newVel.y = 0.0f;
        newVel.z = 0.0f;
    }

    // write to global memory
    nxtPos[4*gid + 0] = newPos.x;
    nxtPos[4*gid + 1] = newPos.y;
    nxtPos[4*gid + 2] = newPos.z;
    nxtPos[4*gid + 3] = newPos.w;

    nxtVel[4*gid + 0] = newVel.x;
    nxtVel[4*gid + 1] = newVel.y;
    nxtVel[4*gid + 2] = newVel.z;
    nxtVel[4*gid + 3] = newVel.w;
}