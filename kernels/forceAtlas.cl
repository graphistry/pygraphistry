#include "common.h"


#define REPULSION_OVERLAP 0.00000001f
#define DEFAULT_NODE_SIZE 0.000001f
#define EPSILON 1.0f // bound whether d(a,b) == 0

#define IS_PREVENT_OVERLAP(flags) (flags & 1)
#define IS_STRONG_GRAVITY(flags) (flags & 2)
#define IS_DISSUADE_HUBS(flags) (flags & 4)
#define IS_LIN_LOG(flags) (flags & 8)


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
        waitEvents[1] = async_work_group_copy(TILEPOINTS2, inDegrees + tileStart, thisTileSize, 0);
        waitEvents[2] = async_work_group_copy(TILEPOINTS3, outDegrees + tileStart, thisTileSize, 0);
		wait_group_events(3, waitEvents);


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
			uint n2Degree = TILEPOINTS2[cachedPoint] + TILEPOINTS3[cachedPoint];
			float2 dist = n1Pos - n2Pos;
			float distance = sqrt(dist.x * dist.x + dist.y * dist.y);
	        int degrees = (n1Degree + 1) * (n2Degree + 1);

	        float force;
	        if (IS_PREVENT_OVERLAP(flags)) {
	            //FIXME Use real sizes: IS_PREVENT_OVERLAP(flags) ? sizes[n1Idx] : 0.0f;
                float n1Size = DEFAULT_NODE_SIZE;
                float n2Size = DEFAULT_NODE_SIZE;
                float distanceB2B = distance - n1Size - n2Size; //border-to-border

	            force = distanceB2B > EPSILON  ? (scalingRatio * degrees / distance) :
	                    distanceB2B < -EPSILON ? (REPULSION_OVERLAP * degrees) :
	                    0.0f;
	        } else {
	            force = scalingRatio * degrees / distance;
	        }

        	n1D += dist * force;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

	}

    const float2 dimensions = (float2) (width, height);
    const float2 centerDist = (dimensions / 2.0f) - n1Pos;

    float gravityForce = gravity * (n1Degree + 1.0f) *
                        (IS_STRONG_GRAVITY(flags) ? sqrt(centerDist.x * centerDist.x + centerDist.y * centerDist.y) : 1.0f);

    outputPositions[n1Idx] =
        n1Pos
        + 0.0001f * centerDist * gravityForce
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
