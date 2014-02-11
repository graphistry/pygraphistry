// The energy with which points repulse each other
#define POINT_REPULSION 0.00001f
// The energy with which the walls repulse points
// #define WALL_REPULSION  1.0f

// The maximum energy we a point can be repulsed by, as a multiple of the base energy.
// So '10' means two points can repulse with a maximum energy of POINT_REPULSION * 10.
// (Normally, as the distance between points approaches 0, the energy approaches infinity. This
// value clamps that.)
#define REPULSION_MAX_MULTIPLE 10.0f

#define RAND_LENGTH 73 //146

// #define EDGE_REPULSION 0.5f

// #define SPRING_LENGTH 0.1f
// #define SPRING_FORCE 0.1f


// Calculate the force of point b on point a, returning a vector indicating the movement to point a
float2 calculatePointForce(float2 a, float2 b, __constant float2* randValues, unsigned int randOffset);


__kernel void nbody_compute_repulsion(
	unsigned int numPoints,
	__global float2* inputPositions,
	__global float2* outputPositions,
	__local float2* tilePoints,
	float2 dimensions,
	__constant float2* randValues,
	unsigned int stepNumber)
{
    dimensions = (float2) (1.0f, 1.0f);
	// use async_work_group_copy() and wait_group_events() to fetch the data from global to local
	// use vloadn() and vstoren() to read/write vectors.

	const unsigned int threadLocalId = (unsigned int) get_local_id(0);
	const unsigned int pointId = (unsigned int) get_global_id(0);

	// The point we're updating
	float2 myPos = inputPositions[pointId];

	// Points per tile = threads per workgroup
	const unsigned int tileSize = (unsigned int) get_local_size(0);
	const unsigned int numTiles = (unsigned int) get_num_groups(0);

	float2 posDelta = (float2) (0.0f, 0.0f);

    unsigned int modulus = numTiles / 7; // tiles per iteration:

	for(unsigned int tile = 0; tile < numTiles; tile++) {

	    if (tile % modulus != stepNumber % modulus) continue;

		const unsigned int tileStart = (tile * tileSize);

		// If numPoints isn't a multiple of tileSize, the last tile will have less than the full
		// number of points. If we detect we'd be reading out-of-bounds data, clamp the number of
		// points we read to be within bounds.
		unsigned int thisTileSize =  tileStart + tileSize < numPoints ?
										tileSize : numPoints - tileStart;

		if(threadLocalId < thisTileSize){
			tilePoints[threadLocalId] = inputPositions[tileStart + threadLocalId];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		for(unsigned int j = 0; j < thisTileSize; j++) {
			unsigned int cachedPoint = j;
			// Don't calculate the forces of a point on itself
			if(tileStart + cachedPoint == pointId) {
				continue;
			}

			float2 otherPoint = tilePoints[cachedPoint];

			posDelta += calculatePointForce(myPos, otherPoint, randValues, stepNumber);
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Calculate force from walls

	myPos += posDelta / clamp(((float) stepNumber)/2.0f, 1.0f, 30.0f);

	// Clamp myPos to be within the walls
	outputPositions[pointId] = clamp(myPos, (float2) (0.0f, 0.0f), dimensions);;

	return;
}


float2 calculatePointForce(float2 a, float2 b, __constant float2* randValues, unsigned int randOffset) {
	float r = (b.x - a.x)*(b.x - a.x) + (b.y - a.y)*(b.y - a.y); //distance(a, b);

	if(r < FLT_EPSILON * FLT_EPSILON) {
		// TODO: We should pass the current tick # into the kernel as an additional source of
		// randomness, then add that to the global id. Right now, the specific random value each
		// point uses is constant. If this point isn't strong enough to get the point 'unstuck'
		// (from, say, a corner,) then it will remain there forever more.
		b = randValues[(get_global_id(0) * randOffset) % RAND_LENGTH];
		r = (b.x - a.x)*(b.x - a.x) + (b.y - a.y)*(b.y - a.y);
	}

	return ((float2) ((b.x - a.x)/r, (b.y - a.y)/r)) * POINT_REPULSION * -1.0f;
}


// __kernel void nbody2d_compute_springs(
// 	unsigned int numEdges,
// 	__global unsigned int* springList,
// 	__global float* springPositions,
// 	__global float* inputPositions,
// 	__global float* outputPositions,
// 	float timeDelta)
// {
// 	// From Hooke's Law, we generally have that the force exerted by a spring is given by
// 	//	F = -k * X, where X is the distance the spring has been displaced from it's natural
// 	// distance, and k is some constant positive real number.
// 	return;
// }


