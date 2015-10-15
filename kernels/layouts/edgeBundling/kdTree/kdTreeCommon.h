// In theory this should be set dynamically, or the code should be rewritten to be
// warp agnostic (as is proper in OpenCL)
// Should be gotten by CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE parameter in the clGetKernelWorkGroupInfo().
// Pretty sure most modern NVidia have warp of 32, and AMD 'wavefront' of 64
// Correctness is guaranteed if WARPSIZE is less than or equal to actual warp size.
#define MAXDEPTH 32

// TODO: I've replaced comparisons >= 0 with > NULLPOINTER for readability.
// We should benchmark to make sure that doesn't impact perf.
#define TREELOCK -2
#define NULLPOINTER -1
