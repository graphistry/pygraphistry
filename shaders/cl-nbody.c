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

__kernel void nbody_kernel_GPU(
    __global float* curPos,
    __global float* curVel,
    int numBodies,
    float deltaTime,
    int epsSqr,
    __local float* localPos,
    __global float* nxtPos,
    __global float* nxtVel)
{
    unsigned int tid = get_local_id(0);
    unsigned int gid = get_global_id(0);
    unsigned int localSize = get_local_size(0);

    // Number of tiles we need to iterate
    unsigned int numTiles = numBodies / localSize;

    // position of this work-item
    float4 myPos = (float4) (curPos[4*gid + 0], curPos[4*gid + 1], curPos[4*gid + 2], curPos[4*gid + 3]);
    float4 acc = (float4) (0.0f, 0.0f, 0.0f, 0.0f);

    for(int i = 0; i < numTiles; ++i)
    {
        // load one tile into local memory
        int idx = i * localSize + tid;
        for(int k=0; k<4; k++)
        {
                localPos[4*tid+k] = curPos[4*idx+k];
        }
        // Synchronize to make sure data is available for processing
        barrier(CLK_LOCAL_MEM_FENCE);
        // calculate acceleration effect due to each body
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
    float4 newPos = myPos + oldVel * deltaTime + acc * 0.5f * deltaTime * deltaTime;
    newPos.w = myPos.w;
    float4 newVel = oldVel + acc * deltaTime;

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