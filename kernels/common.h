//NODECL defined by including file
// (nodecl sets "#define NODECL" for bug https://github.com/Motorola-Mobility/node-webcl/issues/41 )

#ifdef NODECL
	#define TILEPOINTS tilePointsInline
	#define TILEPOINTS2 tilePoints2Inline
	#define TILEPOINTS3 tilePoints3Inline
	#define TILEPOINTS_INLINE_DECL __local float2 tilePointsInline[1000];
	#define TILEPOINTS2_INLINE_DECL __local uint tilePoints2Inline[1000];
	#define TILEPOINTS3_INLINE_DECL __local uint tilePoints3Inline[1000];
#else
	#define TILEPOINTS tilePointsParam
	#define TILEPOINTS2 tilePoints2Param
	#define TILEPOINTS3 tilePoints3Param
	#define TILEPOINTS_INLINE_DECL
	#define TILEPOINTS2_INLINE_DECL
	#define TILEPOINTS3_INLINE_DECL
#endif

#ifdef DEBUG
    // Variadic macros are not supported in OpenCL
    #define debug1(X)     printf(X)
    #define debug2(X,Y)   printf(X,Y)
    #define debug3(X,Y,Z) printf(X,Y,Z)
#else
    #define debug1(X)
    #define debug2(X,Y)
    #define debug3(X,Y,Z)
#endif

// The fraction of tiles to process each execution of this kernel. For example, a value of '10' will
// cause an execution of this kernel to only process every 10th tile.
// The particular subset of tiles is chosen based off of stepNumber.
#define TILES_PER_ITERATION 1

// The length of the 'randValues' array
#define RAND_LENGTH 73 //146
