#define VT 16
#define WARPSIZE 32
#define THREADS 256
#define BINSIZE 256

//******** new ********
// Comments
//*********************
// global_size: power of two & bigger than dataSize
// set workgroup whatever you want (equal to or less than 1024)
// this function supports numBins <= 32, if need it to be a larger number, modify BINSIZE in lane 4
// output: cout,  outputSum: sum,  outputMean: mean, outputMax : max (initialized as nan), outputMin : min (initialized as nan)
// check: is to pick the running function; set 1 as run, 0 as not run. check will be less than or equal to 111111 (binary), 0th bit refers to count, 1st bit
// refers to sum, 2nd bit refers to mean, 3rd bit refers to max, 4th bit refers to min，5th bit refers to constant width
// binStart: the segment of bin, (check >> 5) == 1, binwidths are constant, otherwise binwidths are various
//
// Assumes all outputs are initialized to zero before running kernel.
//

// ******** new ********
// atomic_add_local
// *********************

static void AtomicAddLocal(__local volatile float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((__local volatile unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

// ******** new ********
// atomic_add_global
// *********************

static void AtomicAdd(__global volatile float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((__global volatile unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

// ******** new ********
// atomic_max_local
// *********************

static void AtomicMaxLocal(__local volatile float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = max(prevVal.floatVal, operand);
    } while (atomic_cmpxchg((__local volatile unsigned int*)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

// ******** new ********
// atomic_max_global
// *********************

static void AtomicMax(__global volatile float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = max(prevVal.floatVal, operand);
    } while (atomic_cmpxchg((__global volatile unsigned int*)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

// ******** new ********
// atomic_min_local
// *********************

static void AtomicMinLocal(__local volatile float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = min(prevVal.floatVal, operand);
    } while (atomic_cmpxchg((__local volatile unsigned int*)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

// ******** new ********
// atomic_max_global
// *********************

static void AtomicMin(__global volatile float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = min(prevVal.floatVal, operand);
    } while (atomic_cmpxchg((__global volatile unsigned int*)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}


__kernel void histogram(
        uint numBins, // unsigned int
        uint dataSize, // unsigned int
        __global uint* check, // 1 : count, 1<<1 : sum, 1<<2 : mean, 1<<3 : max, 1<<4 : min, 1<<5: constant binwidth
        __global float* binStart, // bin seg
        __global uint* indices, // indices[dataSize], all elements >= 0 and < data.length
        __global float* data, // data which is indexed into by indices.
        __global volatile int* output, // count: output[numBins]
        __global volatile float* outputSum,  // outputSum[numBins]
        __global float* outputMean, // outputMean[numBins]
        __global volatile float* outputMax, // outputMax[numBins]
        __global volatile float* outputMin // outputMin[numBins]
){

    int gid = get_global_id(0); // BlockIdx * BlockDim + ThreadIdx
    int tid = get_local_id(0); // ThreadIdx
    float step = binStart[1] - binStart[0]; // constant width
    __local int key[THREADS];

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (gid < dataSize) {
        // check average
        if ((*check >> 2) & 1) *check |= 1 | (1 << 1);

        float localElement = data[indices[gid]];

        // check binStart
        if ((*check >> 5) & 1) {
            key[tid] = (int)((localElement - binStart[0]) / step);  // constant width
        } else {
            for (int i = 0; i < numBins; i++) { // various width
                if (localElement >= binStart[i] && localElement < binStart[i + 1]) {
                    key[tid] = i;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // count
        if (*check & 1) {
            __local volatile int histLocal[BINSIZE];
            if (tid < numBins) histLocal[tid] = 0;
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

            atomic_inc(&histLocal[key[tid]]);
            barrier(CLK_LOCAL_MEM_FENCE);

            if (tid < numBins) atomic_add(&output[tid], histLocal[tid]);
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        // sum
        if ((*check >> 1) & 1) {
            __local volatile float sumLocal[BINSIZE];
            if (tid < numBins) sumLocal[tid] = 0.0;
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

            AtomicAddLocal(&sumLocal[key[tid]], localElement);
            barrier(CLK_LOCAL_MEM_FENCE);

            if (tid < numBins) AtomicAdd(&outputSum[tid], sumLocal[tid]);
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        // mean
        if ((*check >> 2) & 1) {
            if (tid < numBins)
            outputMean[tid] =  outputSum[tid]/ (float)output[tid];
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        // max
        if ((*check >> 3) & 1) {
            __local volatile float maxLocal[BINSIZE];
            if (tid < numBins) maxLocal[tid] = outputMax[tid];
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

            AtomicMaxLocal(&maxLocal[key[tid]], localElement);
            barrier(CLK_LOCAL_MEM_FENCE);

            if (tid < numBins) AtomicMax(&outputMax[tid], maxLocal[tid]);
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        // min
        if ((*check >> 4) & 1) {
            __local volatile float minLocal[BINSIZE];
            if (tid < numBins) minLocal[tid] = outputMin[tid];
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

            AtomicMinLocal(&minLocal[key[tid]], localElement);;
            barrier(CLK_LOCAL_MEM_FENCE);

            if (tid < numBins) AtomicMin(&outputMin[tid], minLocal[tid]);
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    return;
}






