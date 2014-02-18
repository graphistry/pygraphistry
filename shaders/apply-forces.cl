// The fraction of tiles to process each execution of this kernel. For example, a value of '10' will
// cause an execution of this kernel to only process every 10th tile.
// The particular subset of tiles is chosen based off of stepNumber.
#define TILES_PER_ITERATION 7

// The length of the 'randValues' array
#define RAND_LENGTH 73 //146

// #define EDGE_REPULSION 0.5f

// #define SPRING_LENGTH 0.1f
// #define SPRING_FORCE 0.1f


// Calculate the force of point b on point a, returning a vector indicating the movement to point a
float2 pointForce(float2 a, float2 b, float force);

// Retrieves a random point from a set of points
float2 randomPoint(__local float2* points, unsigned int numPoints, __constant float2* randValues,
	unsigned int randOffset);

//
__kernel void apply_midpoints(
    unsigned int numPoints,
    unsigned int numSplits,
	__global float2* inputMidPositions,
	__global float2* outputMidPositions,
	__local float2* tilePoints,
	float width,
	float height,
	float charge,
	float gravity,
	__constant float2* randValues,
	unsigned int stepNumber) {


    const float2 dimensions = (float2) (width, height);
	const float alpha = max(0.1f * pown(0.99f, floor(convert_float(stepNumber) / (float) TILES_PER_ITERATION)), 0.005f);  //1.0f / (clamp(((float) stepNumber), 1.0f, 50.0f) + 10.0f);

	const unsigned int threadLocalId = (unsigned int) get_local_id(0);
	const unsigned int pointId = (unsigned int) get_global_id(0);

	float2 myPos = inputMidPositions[pointId];

	const unsigned int tileSize = (unsigned int) get_local_size(0);
	const unsigned int numTiles = (unsigned int) get_num_groups(0);

	float2 posDelta = (float2) (0.0f, 0.0f);

    unsigned int modulus = numTiles / TILES_PER_ITERATION; // tiles per iteration:

	for(unsigned int tile = 0; tile < numTiles; tile++) {

	    if (tile % modulus != stepNumber % modulus) {
	    	continue;
	    }

		const unsigned int tileStart = (tile * tileSize);

		unsigned int thisTileSize =  tileStart + tileSize < numPoints ?
										tileSize : numPoints - tileStart;


		event_t waitEvents[1];
		waitEvents[0] = async_work_group_copy(tilePoints, inputMidPositions + tileStart, thisTileSize, 0);
		wait_group_events(1, waitEvents);

		prefetch(inputMidPositions + ((tile + 1) * tileSize), thisTileSize);

		

		for(unsigned int cachedPoint = 0; cachedPoint < thisTileSize; cachedPoint++) {
			// Don't calculate the forces of a point on itself
			if(tileStart + cachedPoint == pointId) {
				continue;
			}

			float2 otherPoint = tilePoints[cachedPoint];
			if(fast_distance(otherPoint, myPos) <= FLT_EPSILON) {
				otherPoint = randomPoint(tilePoints, thisTileSize, randValues, stepNumber);
			}

			float2 delta = pointForce(myPos, otherPoint, charge * alpha);

			posDelta +=  ((pointId % numSplits) == ((cachedPoint + tileStart) % numSplits)) ? -delta : delta;

		}

		barrier(CLK_LOCAL_MEM_FENCE);		
	}

	float2 center = dimensions / 2.0f;
	posDelta += ((float2) ((center.x - myPos.x), (center.y - myPos.y)) * (gravity * alpha));

	outputMidPositions[pointId] = myPos + posDelta;

	return; 
}    

//Compute elements based on original edges and predefined number of splits in each one
__kernel void apply_midsprings(
	unsigned int numSplits,              // How many times each edge is split (> 0)
	__global uint2* springs,	         // Array of (unsplit) springs, of the form [source node, targer node] (read-only)
	__global uint2* workList, 	         // Array of (unsplit) spring [index, length] pairs to compute (read-only)
	__global float2* inputPoints,        // Current point positions (read-only)
	__global float2* inputMidPoints,     // Current midpoint positions (read-only)
	__global float2* outputMidPoints,    // Point positions after spring forces have been applied (write-only)
	__global float4* springMidPositions, // Positions of the springs after forces are applied. Length
	                                     // len(springs) * 2: one float2 for start, one float2 for
	                                     // end. (write-only)
	float springStrength,                // The rigidity of the springs
	float springDistance,                // The 'at rest' length of a spring
	unsigned int stepNumber
	)
{

    return;
}
    
    

__kernel void apply_points(
	unsigned int numPoints,
	__global float2* inputPositions,
	__global float2* outputPositions,
	__local float2* tilePoints,
	float width,
	float height,
	float charge,
	float gravity,
	__constant float2* randValues,
	unsigned int stepNumber)
{
	// use async_work_group_copy() and wait_group_events() to fetch the data from global to local
	// use vloadn() and vstoren() to read/write vectors.

    const float2 dimensions = (float2) (width, height);

	const float alpha = max(0.1f * pown(0.99f, floor(convert_float(stepNumber) / (float) TILES_PER_ITERATION)), 0.005f);  //1.0f / (clamp(((float) stepNumber), 1.0f, 50.0f) + 10.0f);

	const unsigned int threadLocalId = (unsigned int) get_local_id(0);
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
		waitEvents[0] = async_work_group_copy(tilePoints, inputPositions + tileStart, thisTileSize, 0);
		wait_group_events(1, waitEvents);

		prefetch(inputPositions + ((tile + 1) * tileSize), thisTileSize);


		for(unsigned int cachedPoint = 0; cachedPoint < thisTileSize; cachedPoint++) {
			// Don't calculate the forces of a point on itself
			if(tileStart + cachedPoint == pointId) {
				continue;
			}

			float2 otherPoint = tilePoints[cachedPoint];

			// for(uchar tries = 0; fast_distance(otherPoint, myPos) <= FLT_EPSILON && tries < 100; tries++) {
			if(fast_distance(otherPoint, myPos) <= FLT_EPSILON) {
				otherPoint = randomPoint(tilePoints, thisTileSize, randValues, stepNumber);
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


float2 randomPoint(__local float2* points, unsigned int numPoints, __constant float2* randValues, unsigned int randSeed) {
	// First, we need to get one of the random values from the randValues array, using our randSeed
	const float2 rand2 = randValues[(get_global_id(0) * randSeed) % RAND_LENGTH];
	const float rand = rand2[0] + rand2[1];

	// Now, we need to use the random value to grab one of the points
	const unsigned int pointIndex = convert_uint(numPoints * rand) % numPoints;
	return points[pointIndex];
}


// TODO: Instead of writing out a list
__kernel void apply_springs(
	__global uint2* springs,	       // Array of springs, of the form [source node, targer node] (read-only)
	__global uint2* workList, 	       // Array of spring [index, length] pairs to compute (read-only)
	__global float2* inputPoints,      // Current point positions (read-only)
	__global float2* outputPoints,     // Point positions after spring forces have been applied (write-only)
	__global float4* springPositions,  // Positions of the springs after forces are applied. Length
	                                   // len(springs) * 2: one float2 for start, one float2 for
	                                   // end. (write-only)
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

	const uint springsStart = workList[workItem][0];
	const uint springsCount = workList[workItem][1];

	for(uint curSpringIdx = springsStart; curSpringIdx < springsStart + springsCount; curSpringIdx++) {
		const uint2 curSpring = springs[curSpringIdx];

		float2 source = inputPoints[curSpring[0]];
		float2 target = inputPoints[curSpring[1]];

		float dist = distance(target, source); //sqrt((delta.x * delta.x) + (delta.y * delta.y));
		if(dist > FLT_EPSILON) {
			float force = alpha * springStrength * (dist - springDistance) / dist;
			source += (target - source) * force;
		}
		outputPoints[curSpring[0]] = source;

		// target -= (target - source) * force;
		// outputPoints[curSpring[1]] = (float2) (0.75f, 0.25f);

		springPositions[curSpringIdx] = (float4) (source.x, source.y, target.x, target.y);
	}

	return;
}


