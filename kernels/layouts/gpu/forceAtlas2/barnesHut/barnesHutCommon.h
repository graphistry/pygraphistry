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
#define EPSILON2 0.01f

// In theory this should be set dynamically, or the code should be rewritten to be
// warp agnostic (as is proper in OpenCL)
// Should be gotten by CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE parameter in the clGetKernelWorkGroupInfo().
// Pretty sure most modern NVidia have warp of 32, and AMD 'wavefront' of 64
// Correctness is guaranteed if WARPSIZE is less than or equal to actual warp size.
//#define WARPSIZE 32
#define MAXDEPTH 32

// TODO: I've replaced comparisons >= 0 with > NULLPOINTER for readability.
// We should benchmark to make sure that doesn't impact perf.
#define TREELOCK -2
#define NULLPOINTER -1
