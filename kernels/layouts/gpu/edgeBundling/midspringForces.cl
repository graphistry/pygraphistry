#include "common.h"

//Compute elements based on original edges and predefined number of splits in each one
__kernel void midspringForces(
	unsigned int numSplits,                // 0: How many times each edge is split (> 0)
	const __global uint2* springs,	           // 1: Array of (unsplit) springs, of the form [source node, targer node] (read-only)
	const __global uint4* workList, 	           // 2: Array of (unsplit) spring [index, length] pairs to compute (read-only)
	const __global float2* inputPoints,          // 3: Current point positions (read-only)
    const __global float2* inputForces,          // 4. Forces from point forces
	const __global float2* inputMidPoints,       // 5: Current midpoint positions (read-only)
	__global float2* outputMidPoints,      // 6: Point positions after spring forces have been applied (write-only)
	                                       // len(springs) * 2: one float2 for start, one float2 for
	                                       // end. (write-only)
	float springStrength,                  // 7: The rigidity of the springs
	float springDistance,                  // 8: The 'at rest' length of a spring
	unsigned int stepNumber				   // 9:
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
    const float alpha = max(0.1f * pown(0.85f, stepNumber), 0.01f);
    if (workItem == 0) {
        debug2("Alpha in edge bundling: %f \n", alpha);
    }

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
