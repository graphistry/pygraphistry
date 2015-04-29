/*#define DEBUG*/
#include "common.h"
#undef DEBUG
#include "gsCommon.cl"

// Calculate the force of point b on point a, returning a vector indicating the movement to point a
float2 pointForce(float2 a, float2 b, float force);

// Retrieves a random point from a set of points
float2 randomPoint(__local float2* points, unsigned int numPoints,
                   __constant float2* randValues, unsigned int randOffset);



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
	const __global uint4* workList, 	           // 2: Array of (unsplit) spring [index, length] pairs to compute (read-only)
	const __global float2* inputPoints,          // 3: Current point positions (read-only)
  const __global float2* inputForces,          // 4. Forces from point forces
	const __global float2* inputMidPoints,       // 5: Current midpoint positions (read-only)
	__global float2* outputMidPoints,      // 6: Point positions after spring forces have been applied (write-only)
	__global float4* springMidPositions,   // 7: Positions of the springs after forces are applied. Length
	                                       // len(springs) * 2: one float2 for start, one float2 for
	                                       // end. (write-only)
	__global float4* midSpringColorCoords, // 8: The x,y coordinate to read the edges color from
	float springStrength,                  // 9: The rigidity of the springs
	float springDistance,                  // 10: The 'at rest' length of a spring
	unsigned int stepNumber				   // 11:
)
{

	if (numSplits == 0) return;

  const size_t workItem = (unsigned int) get_global_id(0);
  const uint springsStart = workList[workItem].x;
	const uint springsCount = workList[workItem].y;
	const uint nodeId = workList[workItem].z;

  if (springsCount == 0) {
      return;
  }

  const uint sourceIdx = springs[springsStart].x;
  float2 start = inputPoints[sourceIdx];
  // TODO use a decreasing alpha
	/*const float alpha = max(0.1f * pown(0.99f, floor(convert_float(stepNumber) / (float) TILES_PER_ITERATION)), 0.005f);*/
  const float alpha = 1.0f;

  for (uint curSpringIdx = springsStart; curSpringIdx < springsStart + springsCount; curSpringIdx++) {
    const uint dstIdx = springs[curSpringIdx].y;
    float2 end = inputPoints[dstIdx];
    float thisSpringDist = 0.25f * distance(start, end) / ((float) numSplits + 1.0f);
    thisSpringDist = thisSpringDist;

	  float2 curQP = start;
		uint firstQPIdx = curSpringIdx * numSplits;
		float2 nextQP = inputMidPoints[firstQPIdx];
		float dist = distance(curQP, nextQP);
		float2 nextForce = (dist > FLT_EPSILON) ?  (nextQP - curQP) * alpha * springStrength * (dist - (thisSpringDist)) / (dist)

      : 0.0f;

    for (uint qp = 0; qp < numSplits; qp++) {
      float2 prevQP = curQP;
			float2 prevForce = nextForce;
			curQP = nextQP;
			nextQP = qp < numSplits - 1 ? inputMidPoints[firstQPIdx + qp + 1] : end;
      float dist = distance(curQP, nextQP);
			nextForce = (dist > FLT_EPSILON) ?  (nextQP - curQP) * alpha * springStrength * (dist - (thisSpringDist)) / (dist)
        : 0.0f;
      float2 delta = (qp == numSplits - 1 ? 1.0f : 1.0f) * nextForce - (qp == 0 ? 1.0f : 1.0f) * prevForce;
      outputMidPoints[firstQPIdx + qp] = delta + inputForces[firstQPIdx + qp];
		}
  }
  return;
}
