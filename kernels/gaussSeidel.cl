#include "common.h"

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
