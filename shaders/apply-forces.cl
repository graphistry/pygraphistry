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



//====== FORCE ATLAS 2

#define REPULSION_OVERLAP 0.00000001f

// bound whether d(a,b) == 0
#define EPSILON 1.0f

//set by kernel
//compress booleans into flags
//#define GRAPH_PARAMS_RAW() float scalingRatio, float gravity, unsigned int edgeWeightInfluence, unsigned int flags
//#define GRAPH_PARAMS GRAPH_PARAMS_RAW()
//#define GRAPH_ARGS_RAW scalingRatio, gravity, edgeWeightInfluence, flags
//#define GRAPH_ARGS GRAPH_ARGS_RAW()
#define IS_PREVENT_OVERLAP(flags) (flags & 1)
#define IS_STRONG_GRAVITY(flags) (flags & 2)
#define IS_DISSUADE_HUBS(flags) (flags & 4)
#define IS_LIN_LOG(flags) (flags & 8)

#define DEFAULT_NODE_SIZE 0.000001f


//====================




// The fraction of tiles to process each execution of this kernel. For example, a value of '10' will
// cause an execution of this kernel to only process every 10th tile.
// The particular subset of tiles is chosen based off of stepNumber.
#define TILES_PER_ITERATION 7

// The length of the 'randValues' array
#define RAND_LENGTH 73 //146

// #define EDGE_REPULSION 0.5f

// #define SPRING_LENGTH 0.1f
// #define SPRING_FORCE 0.1f

// BARNES HUT defintions.
// TODO We don't need all these
#define THREADS1 256  /* must be a power of 2 */
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


// Calculate the force of point b on point a, returning a vector indicating the movement to point a
float2 pointForce(float2 a, float2 b, float force);


// Retrieves a random point from a set of points
float2 randomPoint(__local float2* points, unsigned int numPoints, __constant float2* randValues,
	unsigned int randOffset);



__kernel void gaussSeidelMidpoints(
    unsigned int numPoints,
    unsigned int numSplits,
	const __global float2* inputMidPositions,
	__global float2* outputMidPositions,
	__local float2* tilePointsParam,
	float width,
	float height,
	float charge,
	float gravity,
	__constant float2* randValues,
	unsigned int stepNumber) {

    //for debugging: passthrough
    //outputMidPositions[(unsigned int) get_global_id(0)] = inputMidPositions[(unsigned int) get_global_id(0)];
	//return;

    const float2 dimensions = (float2) (width, height);
	const float alpha = max(0.1f * pown(0.99f, floor(convert_float(stepNumber) / (float) TILES_PER_ITERATION)), 0.005f);  //1.0f / (clamp(((float) stepNumber), 1.0f, 50.0f) + 10.0f);

	const unsigned int pointId = (unsigned int) get_global_id(0);

	float2 myPos = inputMidPositions[pointId];

	const unsigned int tileSize = (unsigned int) get_local_size(0);
	const unsigned int numTiles = (unsigned int) get_num_groups(0);

	float2 posDelta = (float2) (0.0f, 0.0f);

    unsigned int modulus = numTiles / TILES_PER_ITERATION; // tiles per iteration:


	TILEPOINTS_INLINE_DECL;

	for(unsigned int tile = 0; tile < numTiles; tile++) {

	    if (tile % modulus != stepNumber % modulus) {
	    	continue;
	    }

		const unsigned int tileStart = (tile * tileSize);

		unsigned int thisTileSize =  tileStart + tileSize < numPoints ?
										tileSize : numPoints - tileStart;

		//block on fetching current tile
		event_t waitEvents[1];
		waitEvents[0] = async_work_group_copy(TILEPOINTS, inputMidPositions + tileStart, thisTileSize, 0);
		wait_group_events(1, waitEvents);

		//hint fetch of next tile
		prefetch(inputMidPositions + ((tile + 1) * tileSize), thisTileSize);

		for(unsigned int cachedPoint = 0; cachedPoint < thisTileSize; cachedPoint++) {
			// Don't calculate the forces of a point on itself
			if (tileStart + cachedPoint == pointId) {
				continue;
			}

			if((pointId % numSplits) != ((cachedPoint + tileStart) % numSplits)) {
				continue;
			}

			float2 otherPoint = TILEPOINTS[cachedPoint];
			float err = fast_distance(otherPoint, myPos);
			if (err <= FLT_EPSILON) {
				otherPoint = randomPoint(TILEPOINTS, thisTileSize, randValues, stepNumber);
			}

			posDelta += (err <= FLT_EPSILON ? 0.1f : 1.0f) * pointForce(myPos, otherPoint, charge * alpha) * -1;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	float2 center = dimensions / 2.0f;
	posDelta += ((float2) ((center.x - myPos.x), (center.y - myPos.y)) * (gravity * alpha));

	outputMidPositions[pointId] = myPos + posDelta;

	return;
}


//Compute elements based on original edges and predefined number of splits in each one
__kernel void gaussSeidelMidsprings(
	unsigned int numSplits,                // 0: How many times each edge is split (> 0)
	const __global uint2* springs,	           // 1: Array of (unsplit) springs, of the form [source node, targer node] (read-only)
	const __global uint2* workList, 	           // 2: Array of (unsplit) spring [index, length] pairs to compute (read-only)
	const __global float2* inputPoints,          // 3: Current point positions (read-only)
	const __global float2* inputMidPoints,       // 4: Current midpoint positions (read-only)
	__global float2* outputMidPoints,      // 5: Point positions after spring forces have been applied (write-only)
	__global float4* springMidPositions,   // 6: Positions of the springs after forces are applied. Length
	                                       // len(springs) * 2: one float2 for start, one float2 for
	                                       // end. (write-only)
	__global float4* midSpringColorCoords, // 7: The x,y coordinate to read the edges color from
	float springStrength,                  // 8: The rigidity of the springs
	float springDistance,                  // 9: The 'at rest' length of a spring
	unsigned int stepNumber				   // 10:
)
{

	if (numSplits == 0) return;

    const size_t workItem = (unsigned int) get_global_id(0);
    const uint springsStart = workList[workItem].x;
	const uint springsCount = workList[workItem].y;
    const uint sourceIdx = springs[springsStart].x;
    float2 start = inputPoints[sourceIdx];

	const float alpha = max(0.1f * pown(0.99f, floor(convert_float(stepNumber) / (float) TILES_PER_ITERATION)), 0.005f);

    for (uint curSpringIdx = springsStart; curSpringIdx < springsStart + springsCount; curSpringIdx++) {

		float2 curQP = start;
		uint firstQPIdx = curSpringIdx * numSplits;
		float2 nextQP = inputMidPoints[firstQPIdx];
		float dist = distance(curQP, nextQP);
		float2 nextForce = (dist > FLT_EPSILON) ?
		    -1.0f * (curQP - nextQP) * alpha * springStrength * (dist - springDistance) / dist
		    : 0.0f;

        for (uint qp = 0; qp < numSplits; qp++) {
        	// Set the color coordinate for this mid-spring to the coordinate of the start point
        	midSpringColorCoords[curSpringIdx * (numSplits + 1) + qp] = (float4)(start, start);

			float2 prevQP = curQP;
			float2 prevForce = nextForce;
			curQP = nextQP;
			nextQP = qp < numSplits - 1 ? inputMidPoints[firstQPIdx + qp + 1] : inputPoints[springs[curSpringIdx].y];
			nextForce = (dist > FLT_EPSILON) ?
		        (nextQP - curQP) * alpha * springStrength * (dist - springDistance) / dist
		        : 0.0f;
		    float2 delta = (qp == numSplits - 1 ? 1.0f : 1.0f) * nextForce - (qp == 0 ? 1.0f : 1.0f) * prevForce;
		    outputMidPoints[firstQPIdx + qp] = curQP + delta;
		    springMidPositions[curSpringIdx * (numSplits + 1) + qp] = (float4) (prevQP.x, prevQP.y, curQP.x, curQP.y);
		}
        const uint dstIdx = springs[curSpringIdx].y;
	    float2 end = inputPoints[dstIdx];
		springMidPositions[(curSpringIdx + 1) * (numSplits + 1) - 1] = (float4) (curQP.x, curQP.y, end.x, end.y);
		midSpringColorCoords[(curSpringIdx + 1) * (numSplits + 1) - 1] = (float4) (start, start);

    }

    return;
}

__kernel void gaussSeidelPoints(
	unsigned int numPoints,
	const __global float2* inputPositions,
	__global float2* outputPositions,
	__local float2* tilePointsParam, //FIXME make nodecl accept local params
	float width,
	float height,
	float charge,
	float gravity,
	__constant float2* randValues,
	unsigned int stepNumber)
{

	// use async_work_group_copy() and wait_group_events() to fetch the data from global to local
	// use vloadn() and vstoren() to read/write vectors.

	TILEPOINTS_INLINE_DECL;

    const float2 dimensions = (float2) (width, height);

	const float alpha = max(0.1f * pown(0.99f, floor(convert_float(stepNumber) / (float) TILES_PER_ITERATION)), 0.005f);  //1.0f / (clamp(((float) stepNumber), 1.0f, 50.0f) + 10.0f);

	const unsigned int pointId = (unsigned int) get_global_id(0);

	// The point we're updating
	float2 myPos = inputPositions[pointId];

	// Points per tile = threads per workgroup
	const unsigned int tileSize = (unsigned int) get_local_size(0);
	const unsigned int numTiles = (unsigned int) get_num_groups(0);

	float2 posDelta = (float2) (0.0f, 0.0f);

    unsigned int modulus = numTiles / TILES_PER_ITERATION; // tiles per iteration:


	for(unsigned int tile = 0; tile < numTiles; tile++) {

	    if (tile % modulus != stepNumber % modulus) {
	    	continue;
	    }

		const unsigned int tileStart = (tile * tileSize);

		// If numPoints isn't a multiple of tileSize, the last tile will have less than the full
		// number of points. If we detect we'd be reading out-of-bounds data, clamp the number of
		// points we read to be within bounds.
		unsigned int thisTileSize =  tileStart + tileSize < numPoints ?
										tileSize : numPoints - tileStart;


		// if(threadLocalId < thisTileSize){
		// 	tilePoints[threadLocalId] = inputPositions[tileStart + threadLocalId];
		// }

		// barrier(CLK_LOCAL_MEM_FENCE);


		event_t waitEvents[1];

		//continue; //FIXME continuing loop from here busts code if tilePoints is dynamic param

		waitEvents[0] = async_work_group_copy(TILEPOINTS, inputPositions + tileStart, thisTileSize, 0);
		wait_group_events(1, waitEvents);
		prefetch(inputPositions + ((tile + 1) * tileSize), thisTileSize);


		for(unsigned int cachedPoint = 0; cachedPoint < thisTileSize; cachedPoint++) {
			// Don't calculate the forces of a point on itself
			if(tileStart + cachedPoint == pointId) {
				continue;
			}

			float2 otherPoint = TILEPOINTS[cachedPoint];

			// for(uchar tries = 0; fast_distance(otherPoint, myPos) <= FLT_EPSILON && tries < 100; tries++) {
			if(fast_distance(otherPoint, myPos) <= FLT_EPSILON) {
				otherPoint = randomPoint(TILEPOINTS, thisTileSize, randValues, stepNumber);
			}

			posDelta += pointForce(myPos, otherPoint, charge * alpha);
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Force of gravity pulling the points toward the center
	float2 center = dimensions / 2.0f;
	// TODO: Should we be dividing the stength of gravity by TILES_PER_ITERATION? We only consider
	// 1 / TILES_PER_ITERATION of the total points in any execution, but here we apply full gravity.
	posDelta += ((float2) ((center.x - myPos.x), (center.y - myPos.y)) * (gravity * alpha));


	// Clamp myPos to be within the walls
	// outputPositions[pointId] = clamp(myPos + posDelta, (float2) (0.0f, 0.0f), dimensions);

	outputPositions[pointId] = myPos + posDelta;

	return;
}


float2 pointForce(float2 a, float2 b, float force) {
	const float2 d = (float2) ((b.x - a.x), (b.y - a.y));
	// k = force / distance^2
	const float k = force / max((d.x * d.x) + (d.y * d.y), FLT_EPSILON);

	return (float2) (d.x * k, d.y * k);
}

float2 randomPoint(__local float2* points, unsigned int numPoints, __constant float2* randValues, unsigned int randOffset) {
	// First, we need to get one of the random values from the randValues array, using our randSeed
	const float2 rand2 = randValues[(get_global_id(0) * randOffset) % RAND_LENGTH];
	const float rand = rand2.x + rand2.y;

	// // Now, we need to use the random value to grab one of the points
	const unsigned int pointIndex = convert_uint(numPoints * rand) % numPoints;
	return points[pointIndex];
}


//for each edge source, find corresponding point and tension from destination points
__kernel void gaussSeidelSprings(
	const __global uint2* springs,	       // Array of springs, of the form [source node, target node] (read-only)
	const __global uint2* workList, 	       // Array of spring [source index, sinks length] pairs to compute (read-only)
	const __global float2* inputPoints,      // Current point positions (read-only)
	__global float2* outputPoints,     // Point positions after spring forces have been applied (write-only)
	float springStrength,              // The rigidity of the springs
	float springDistance,              // The 'at rest' length of a spring
	unsigned int stepNumber
	)
{
	const float alpha = max(0.1f * pown(0.99f, floor(convert_float(stepNumber) / (float) TILES_PER_ITERATION)), 0.005f);
	// const float alpha = max(0.1f * pown(0.99f, stepNumber), FLT_EPSILON * 2.0f);

	// From Hooke's Law, we generally have that the force exerted by a spring is given by
	//	F = -k * X, where X is the distance the spring has been displaced from it's natural
	// distance, and k is some constant positive real number.

	// d = target - source;
	// l1 = Math.sqrt(distance^2);
	// l = alpha * strengths[i] * ((l1) - distances[i]) / l1;
	// distance *= l;
	// k = source.weight / (target.weight + source.weight)
	// target -= distance * k;
	// k = 1 - k;
	// source += distance * k;

	const size_t workItem = (unsigned int) get_global_id(0);

	const uint springsStart = workList[workItem].x;
	const uint springsCount = workList[workItem].y;

    const uint sourceIdx = springs[springsStart].x;

	float2 source = inputPoints[sourceIdx];
	for(uint curSpringIdx = springsStart; curSpringIdx < springsStart + springsCount; curSpringIdx++) {
		const uint2 curSpring = springs[curSpringIdx];
		float2 target = inputPoints[curSpring.y];
		float dist = distance(target, source); //sqrt((delta.x * delta.x) + (delta.y * delta.y));
		if(dist > FLT_EPSILON) {
			float force = alpha * springStrength * (dist - springDistance) / dist;
			source += (target - source) * force;
		}
	}
	outputPoints[sourceIdx] = source;
}


__kernel void gaussSeidelSpringsGather(
	const __global uint2* springs,	       // Array of springs, of the form [source node, target node] (read-only)
	const __global uint2* workList, 	       // Array of spring [source index, sinks length] pairs to compute (read-only)
	const __global float2* inputPoints,      // Current point positions (read-only)
	__global float4* springPositions   // Positions of the springs after forces are applied. Length
	                                   // len(springs) * 2: one float2 for start, one float2 for
	                                   // end. (write-only)
	)
{

	const size_t workItem = (unsigned int) get_global_id(0);
	const uint springsStart = workList[workItem].x;
	const uint springsCount = workList[workItem].y;

    const uint sourceIdx = springs[springsStart].x;
	const float2 source = inputPoints[sourceIdx];

	for (uint curSpringIdx = springsStart; curSpringIdx < springsStart + springsCount; curSpringIdx++) {
		const uint2 curSpring = springs[curSpringIdx];
		const float2 target = inputPoints[curSpring.y];
		springPositions[curSpringIdx] = (float4) (source.x, source.y, target.x, target.y);
	}

}


//========== FORCE ATLAS 2

//repulse points and apply gravity
__kernel void forceAtlasPoints (
	//input

	//GRAPH_PARAMS
    float scalingRatio, float gravity, unsigned int edgeWeightInfluence, unsigned int flags,

	__local float2* tilePointsParam, //FIXME make nodecl accept local params
	__local uint* tilePoints2Param, //FIXME make nodecl accept local params
	__local uint* tilePoints3Param, //FIXME make nodecl accept local params
	unsigned int numPoints,
	const __global float2* inputPositions,
	float width,
	float height,
	unsigned int stepNumber,
	const __global uint* inDegrees,
	const __global uint* outDegrees,

	//output
	__global float2* outputPositions
) {

    const unsigned int n1Idx = (unsigned int) get_global_id(0);
    const unsigned int tileSize = (unsigned int) get_local_size(0);
    const unsigned int numTiles = (unsigned int) get_num_groups(0);
    unsigned int modulus = numTiles / TILES_PER_ITERATION; // tiles per iteration:

	TILEPOINTS_INLINE_DECL;
	TILEPOINTS2_INLINE_DECL;
	TILEPOINTS3_INLINE_DECL;


    float2 n1Pos = inputPositions[n1Idx];
    float2 n1D = (float2) (0.0f, 0.0f);

    uint n1Degree = inDegrees[n1Idx] + outDegrees[n1Idx];


    //FIXME IS_PREVENT_OVERLAP(flags) ? sizes[n1Idx] : 0.0f;
    float n1Size = DEFAULT_NODE_SIZE;

    for(unsigned int tile = 0; tile < numTiles; tile++) {
        if (tile % modulus != stepNumber % modulus) {
            continue;
        }


		const unsigned int tileStart = (tile * tileSize);
		unsigned int thisTileSize =  tileStart + tileSize < numPoints ?
										tileSize : numPoints - tileStart;

		//block on fetching current tile

		event_t waitEvents[3];
		waitEvents[0] = async_work_group_copy(TILEPOINTS, inputPositions + tileStart, thisTileSize, 0);
		if (IS_PREVENT_OVERLAP(flags)) {
			waitEvents[1] = async_work_group_copy(TILEPOINTS2, inDegrees + tileStart, thisTileSize, 0);
			waitEvents[2] = async_work_group_copy(TILEPOINTS3, outDegrees + tileStart, thisTileSize, 0);
		}
		wait_group_events(IS_PREVENT_OVERLAP(flags) ? 3 : 1, waitEvents);


		//hint fetch of next tile
		prefetch(inputPositions + ((tile + 1) * tileSize), thisTileSize);
		prefetch(inDegrees + ((tile + 1) * tileSize), thisTileSize);
		prefetch(outDegrees + ((tile + 1) * tileSize), thisTileSize);

		for(unsigned int cachedPoint = 0; cachedPoint < thisTileSize; cachedPoint++) {
			// Don't calculate the forces of a point on itself
			if (tileStart + cachedPoint == n1Idx) {
				continue;
			}

			float2 n2Pos = TILEPOINTS[cachedPoint];
			float2 dist = n1Pos - n2Pos;
			float distanceSqr = dist.x * dist.x + dist.y * dist.y;

			//FIXME include in prefetch etc.
	        float n2Size = DEFAULT_NODE_SIZE; //graphSettings->isPreventOverlap ? sizes[n2Idx] : 0.0f;
	        uint n2Idx = tileStart + cachedPoint;
	        uint n2Degree = IS_PREVENT_OVERLAP(flags) ? TILEPOINTS2[cachedPoint] + TILEPOINTS3[cachedPoint] : 0;

	        float force;
	        if (IS_PREVENT_OVERLAP(flags)) {

	            //FIXME only apply after convergence <-- use stepNumber?

	            //border-to-border approximation
	            float distance = sqrt(distanceSqr) - n1Size - n2Size;
	            int degrees = (n1Degree + 1) * (n2Degree + 1);

	            force =
	                  distance > EPSILON    ? (scalingRatio * degrees / distance)
	                : distance < -EPSILON   ? (REPULSION_OVERLAP * degrees)
	                : 0.0f;

	        } else {
	            force = scalingRatio / distanceSqr;
	        }

        	n1D += dist * force;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

	}

	//FIXME use mass
	//FIXME gravity relative to width/height center?

    const float2 dimensions = (float2) (width, height);
    const float2 centerDist = (dimensions/2.0f) - n1Pos;

    float gravityForce =
        1.0f //mass
        * gravity
        * (n1Degree + 1.0f)
        / (IS_STRONG_GRAVITY(flags) ? 1.0f : sqrt(centerDist.x * centerDist.x + centerDist.y * centerDist.y));


    outputPositions[n1Idx] =
    	n1Pos
    	+ 0.01f * centerDist * gravityForce
    	+ 0.00001f * n1D;

	return;
}


//attract edges and apply forces
__kernel void forceAtlasEdges(
    //input

    //GRAPH_PARAMS
    float scalingRatio, float gravity, unsigned int edgeWeightInfluence, unsigned int flags,

	const __global uint2* springs,	       	// Array of springs, of the form [source node, target node] (read-only)
	const __global uint2* workList, 	       	// Array of spring [source index, sinks length] pairs to compute (read-only)
	const __global float2* inputPoints,      	// Current point positions (read-only)
	unsigned int stepNumber,

	//output
	__global float2* outputPoints
) {

    const size_t workItem = (unsigned int) get_global_id(0);
    const uint springsStart = workList[workItem].x;
    const uint springsCount = workList[workItem].y;
    const uint sourceIdx = springs[springsStart].x;

    //====== attact edges

    float2 n1Pos = inputPoints[sourceIdx];

    //FIXME IS_PREVENT_OVERLAP(flags) ? sizes[n1Idx] : 0.0f;
    float n1Size = DEFAULT_NODE_SIZE;

    //FIXME start with previous deriv?
    float2 n1D = (float2) (0.0f, 0.0f);

	for(uint curSpringIdx = springsStart; curSpringIdx < springsStart + springsCount; curSpringIdx++) {

        const uint2 curSpring = springs[curSpringIdx];

        float2 n2Pos = inputPoints[curSpring.y];

        //FIXME from param
        float n2Size = DEFAULT_NODE_SIZE; //graphSettings->isPreventOverlap ? sizes[curSpring[1]] : 0.0f;
        uint wMode = edgeWeightInfluence;
        float weight = 1.0f; //wMode ? edgeWeight[curSpringIdx] : 0.0f;

        float weightMultiplier =
            wMode == 0      ? 1.0f
            : wMode == 1    ? weight
            : pown(weight, wMode);

        float2 dist = n2Pos - n1Pos;

        float distance =
            sqrt(dist.x * dist.x + dist.y * dist.y)
            - (IS_PREVENT_OVERLAP(flags) ? n1Size + n2Size : 0.0f);

        float force =
            (IS_PREVENT_OVERLAP(flags) && distance < EPSILON)
                ? 0.0f
                : (weightMultiplier
                    * (IS_LIN_LOG(flags) ? log(1.0f + 100.0f * distance) : distance)
                    / (IS_DISSUADE_HUBS(flags) ? springsCount + 1.0f : 1.0f));

        n1D += dist * force;
    }

    //====== apply

    float2 source = n1Pos+ n1D * 0.0001f;

    outputPoints[sourceIdx] = source;

    return;

}


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
  __global volatile int* maxdepthd
  ) {
  x_cords[0] = 1 /0;
  size_t gid = get_global_id(0);
  size_t global_size = get_global_size(0);
  for (int i = gid; i < numPoints; i += global_size) {
    x_cords[i] = inputPositions[i][0];
    y_cords[i] = inputPositions[i][1];
    mass[i] = 1.0f; //1.0f;
  }
  if (gid == 0) {
    *maxdepthd = -1;
    *blocked = 0;
  }
}


//__attribute__ ((reqd_work_group_size(THREADS1, 1, 1)))
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
    const int num_bodies,
    const int num_nodes)
{

  size_t tid = get_local_id(0);
  size_t gid = get_group_id(0);
  size_t dim = get_local_size(0);
  size_t global_dim_size = get_global_size(0);
  size_t idx = get_global_id(0);

  float minx, maxx, miny, maxy;
  __local float sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1], smaxy[THREADS1];
  minx = maxx = x_cords[0];
  miny = maxy = y_cords[0];
  float val;
  int inc = global_dim_size;
  for (int j = idx; j < num_bodies; j += inc) {
    val = x_cords[j];
    minx = min(val, minx);
    maxx = max(val, maxx);
    val = y_cords[j];
    miny = min(val, miny);
    maxy = max(val, maxy);
  }
  sminx[tid] = minx;
  smaxx[tid] = maxx;
  sminy[tid] = miny;
  smaxy[tid] = maxy;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int step = (dim / 2); step > 0; step = step / 2) {
    if (tid < step) {
    //printf("smin: %f\n", sminx);
    sminx[tid] = minx = min(sminx[tid] , sminx[tid + step]);
    smaxx[tid] = maxx = max(smaxx[tid], smaxx[tid + step]);
    sminy[tid] = miny = min(sminy[tid] , sminy[tid + step]);
    smaxy[tid] = maxy = max(smaxy[tid], smaxy[tid + step]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  /*printf("minx: %f \n", minx);*/
  /*printf("maxx: %f \n", maxx);*/

  // Only one thread needs to outdate the global buffer
  inc = (global_dim_size / dim) - 1;
  if (tid == 0) {
    global_x_mins[gid] = minx;
    global_x_maxs[gid] = maxx;
    global_y_mins[gid] = miny;
    global_y_maxs[gid] = maxy;
    inc = (global_dim_size / dim) - 1;
    if (inc == atomic_inc(blocked)) {
      for(int j = 0; j <= inc; j++) {
        minx = min(minx, global_x_mins[j]);
        maxx = max(maxx, global_x_maxs[j]);
        miny = min(miny, global_y_mins[j]);
        maxy = max(maxy, global_y_maxs[j]);
      }

      // Compute the radius
      val = max(maxx - minx, maxy - miny);
      *radiusd = (float) (val* 0.5f);

      int k = num_nodes;
      *bottomd = k;
      // TODO bottomd;

      printf("k: %d \k", k);
      mass[k] = -1.0f;
      start[k] = 0;



      x_cords[num_nodes] = (minx + maxx) * 0.5f;
      y_cords[num_nodes] = (miny + maxy) * 0.5f;
      k *= 4;
      printf(" here k: %d  \n", k);
      for (int i = 0; i < 4; i++) children[k + i] = -1;
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
    const int num_bodies,
    const int num_nodes) {

  float radius = *radiusd;
  //printf("Readius : %f", radius);
  float rootx = x_cords[num_nodes];
  float rooty = y_cords[num_nodes];
  /*printf("rootx: %f \n", rootx);*/
  /*printf("rooty: %f \n", rooty);*/
  //printf("Bottom: %d, num_bodies: %d", *bottom, num_bodies);
  float r;
  int localmaxdepth = 1;
  int skip = 1;
  int inc =  get_global_size(0);
  int i = get_global_id(0);
  float x, y;
  int j;
  float px, py;
  int ch, n, cell, locked, patch;
  int depth;
  while (i < num_bodies) {
    if (skip != 0) {
      skip = 0;
      px = x_cords[i];
      py = y_cords[i];
      n = num_nodes;
      depth = 1;
      r = radius;
      j = 0;
      if (rootx < px) j = 1;
      if (rooty < py) j += 2;
    }
    ch = child[n*4 + j];

    while (ch >= num_bodies) {
      n = ch;
      depth++;
      r *= 0.5f;
      j = 0;
      // determine which child to follow
      if (x_cords[n] < px) j = 1;
      if (y_cords[n] < py) j += 2;
      ch = child[n*4+j];
      //printf("ch: %d \n", ch);
    }

    if (ch != -2 ) {
    locked = n*4+j;
    //int test = child[locked];
    //mem_fence(CLK_GLOBAL_MEM_FENCE);
    // return;
    if (ch == atomic_cmpxchg(&child[locked], ch, -2)) {
      //mem_fence(CLK_GLOBAL_MEM_FENCE);

      if(ch == -1) {
        child[locked] = i;
      } else {
        patch = -1;
        // create new cell(s) and insert the old and new body
        int test = 1000000;
          //printf("Child: %d \n", ch);
        do {
          depth++;
          /*printf("*bottom : %d\n", *bottom);*/
          cell = atomic_dec(bottom) - 1;
          //printf("Cell: %d \n", cell);

          if (cell <= num_bodies) {
            printf("\nI:  %d\n", i);
            printf("Error\n");
            printf("x_cords[ch]: %f \n", x_cords[ch]);
            printf("y_cords[ch]: %f \n", y_cords[ch]);
            printf("x : %f \n", px);
            printf("y : %f \n", py);
            printf("r: %f \n", r);
            return;
            break;
            *bottom = num_nodes;
            // TODO (paden) This will break if it goes over!
            break;
            return;
            /*return;*/
             /*return;*/
          }
          patch = max(patch, cell);

          x = (j & 1) * r;
          y = ((j >> 1) & 1) * r;
          r *= 0.5f;

          mass[cell] = -1.0f;
          start[cell] = -1;
          x = x_cords[cell] = x_cords[n] - r + x;
          y = y_cords[cell] = y_cords[n] - r + y;
          for (int k = 0; k < 4; k++) child[cell*4+k] = -1;

          if (patch != cell) {
            child[n*4+j] = cell;
          }

          j = 0;
          if (x < x_cords[ch]) j = 1;
          if (y < y_cords[ch]) j += 2;
          /*printf("x_cords[ch]: %f \n", x_cords[ch]);*/
          /*printf("y_cords[ch]: %f \n", y_cords[ch]);*/
          /*printf("px : %f \n", px);*/
          /*printf("py : %f \n", py);*/
          child[cell*4+j] = ch;
          if (x_cords[ch] == px && y_cords[ch] == py) {
            x_cords[ch] += 0.00001;
            y_cords[ch] += 0.00001;
          }

          n = cell;
          j = 0;
          if (x < px) j = 1;
          if (y < py) j += 2;

          ch = child[n*4+j];
          test--;
        } while (test > 0 && ch >= 0);
        child[n*4+j] = i;
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        child[locked] = patch;
       }
        localmaxdepth = max(depth, localmaxdepth);
        i += inc;  // move on to next body
        skip = 1;
      }
      }
  }
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
    const int num_bodies,
    const int num_nodes) {
  int i, j, k, inc, num_children_missing, cnt, bottom_value, child;
  float m, cm, px, py;
  // TODO change this to THREAD3 Why?
  volatile int missing_children[THREADS1 * 4];
  // TODO chache kernel information

  bottom_value = *bottom;
  //printf("bottom value: %d \n", bottom_value);
  inc = get_global_size(0);
  // Align work to WARP SIZE
  k = (bottom_value & (-WARPSIZE)) + get_global_id(0);
  if (k < bottom_value) k += inc;

  num_children_missing = 0;

  while (k <= num_nodes) {
    if (num_children_missing == 0) { // Must be new cell
      cm = 0.0f;
      px = 0.0f;
      py = 0.0f;
      cnt = 0;
      j = 0;
      for (i = 0; i < 4; i++) {
        child = children[k*4+i];
        if (child >= 0) {
          if (i != j) {
            // Moving children to front. Apparently needed later
            // TODO figure out why this is
            children[k*4+i] = -1;
            children[k*4+j] = child;
          }
          missing_children[num_children_missing*THREADS1+get_local_id(0)] = child;
          m = mass[child];
          num_children_missing++;
          if (m >= 0.0f) {
            // Child has already been touched
            num_children_missing--;
            if (child >= num_bodies) { // Count the bodies. TODO Why?
              cnt += count[child] - 1;
            }
            // Sum mass and positions
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
    const int num_bodies,
    const int num_nodes) {
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
              start[child] = start_index;
              start_index += count[child];
            } else if (child >= 0) {
              sort[start_index] = child;
              start_index++;
            }
          }
          k -= decrement;
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
    /*printf("allBlock[warp]: %d warp %d \n", allBlock[warpId], warpId);*/
    allBlock[warpId] = old;

    //printf("Return : %d \n", );
    return ret;
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
    const int num_bodies,
    const int num_nodes) {
  int idx = get_global_id(0);
  int k, index, i;
  int warp_id, starting_warp_thread_id, shared_mem_offset, difference, depth, child;
  __local volatile int child_index[MAXDEPTH * THREADS1/WARPSIZE], parent_index[MAXDEPTH * THREADS1/WARPSIZE];
   __local volatile int allBlock[THREADS1 / WARPSIZE];
  __local volatile float dq[MAXDEPTH * THREADS1/WARPSIZE];
  __local volatile int shared_step, shared_maxdepth;
  __local volatile int allBlocks[THREADS1/WARPSIZE];
  float px, py, ax, ay, dx, dy, temp;
  int global_size = get_global_size(0);
  //printf("Radius %f \n", *radiusd);
   /*printf("num nodes: %d", num_nodes);*/
   /*printf("num bodies; %d", num_bodies);*/

  if (get_local_id(0) == 0) {
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
      //temp =  1/0;
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
  for (k = idx; k < num_bodies; k+=global_size) {
    //atomic_add(&allBlock[warp_id], 1);
    index = sort[k];
    px = x_cords[index];
    py = y_cords[index];
    ax = 0.0f;
    ay = 0.0f;
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
          dx = x_cords[child] - px;
          dy = y_cords[child] - py;
          temp = dx*dx + (dy*dy + 0.0001f);
          if ((child < num_bodies)  ||  thread_vote(allBlocks, warp_id, temp >= dq[depth]) )  {
            temp = native_rsqrt(temp);
            temp = mass[child] * temp * temp *temp;
            ax += dx * temp;
            ay += dy * temp;
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
    const int num_bodies,
    const int num_nodes) {
    const float dtime = 0.025f;
    const float dthf = dtime * 0.5f;
    float velx, vely;

    int inc = get_global_size(0);
    for (int i = get_global_id(0); i < num_bodies; i+= inc) {
      velx = accx[i] * dthf;
      vely = accy[i] * dthf;

      x_cords[i] += velx * dtime;
      y_cords[i] += vely * dtime;
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
  __global volatile int* maxdepthd
  ) {
  size_t gid = get_global_id(0);
  size_t global_size = get_global_size(0);
  for (int i = gid; i < numPoints; i += global_size) {
    outputPositions[i][0] = x_cords[i];
    outputPositions[i][1] = y_cords[i];
    printf("x_cords[i] = %f \n", x_cords[i]);
  }
}





