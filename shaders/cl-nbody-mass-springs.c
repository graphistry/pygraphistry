// Number of elements per 2d point
#define COMPONENTS_2D 2

// The nominal width of the graph. The graph size (as defined by the walls) is scaled to this for
// computation. (Reason: forces are calculated using the inverse of the (distance squared). If the
// actual width of the graph is < 1.0, then all forces will always be > the base enegery, e.g.
// POINT_REPULSION. We rarely want that, so we scale distances to be a fraction of GRAPH_WIDTH.)
#define GRAPH_WIDTH 1000.0f

// The energy with which points repulse each other
#define POINT_REPULSION 0.01f
// The energy with which the walls repulse points
// #define WALL_REPULSION  1.0f
// The maximum energy we a point can be repulsed by, as a multiple of the base energy.
// So '10' means two points can repulse with a maximum energy of POINT_REPULSION * 10.
// (Normally, as the distance between points approaches 0, the energy approaches infinity. This
// value clamps that.)
#define REPULSION_MAX_MULTIPLE 10.0f

// #define EDGE_REPULSION 0.5f

// #define SPRING_LENGTH 0.1f
// #define SPRING_FORCE 0.1f

// TODO: Add in a repulsive force from the four walls
// TODO: Allow the wall coordinates to be passed as a kernel argument


// Calculate the force of point b on point a, returning a vector indicating the movement to point a
float2 calculatePointForce(float2 a, float2 b);


__kernel void nbody_compute_repulsion(
	unsigned int numPoints,
	__global float* inputPositions,
	__global float* outputPositions,
	float timeDelta,
	__local float* tilePoints)
{
	// use async_work_group_copy() and wait_group_events() to fetch the data from global to local
	// use vloadn() and vstoren() to read/write vectors.

	// (left, right, bottom, top)
	const float4 walls = (float4) (0.05f, 0.95f, 0.05f, 0.95f);
	// const float2 w = (float2) (distance(walls.xz, walls.yz), distance(walls.xz, walls.xw)) * GRAPH_WIDTH;
	const float2 w = GRAPH_WIDTH / (float2) (walls.y - walls.x, walls.w - walls.z);

	const unsigned int threadLocalId = (unsigned int) get_local_id(0);
	const unsigned int pointId = (unsigned int) get_global_id(0) * COMPONENTS_2D;

	// The point we're updating
	float2 myPos = (float2) (inputPositions[pointId + 0], inputPositions[pointId + 1]);

	// Points per tile = threads per workgroup
	const unsigned int tileSize = (unsigned int) get_local_size(0);
	const unsigned int numTiles = max(numPoints / tileSize, (uint) 1);

	float2 posDelta = (float2) (0.0f, 0.0f);

	for(unsigned int tile = 0; tile < numTiles; tile++) {
		const unsigned int tileStart = (tile * (tileSize * COMPONENTS_2D));
		const unsigned int tilePointId = tileStart + (threadLocalId * COMPONENTS_2D);

		tilePoints[(COMPONENTS_2D * threadLocalId) + 0] = inputPositions[tilePointId + 0];
		tilePoints[(COMPONENTS_2D * threadLocalId) + 1] = inputPositions[tilePointId + 1];

		barrier(CLK_LOCAL_MEM_FENCE);

		for(unsigned int j = 0; j < tileSize; j++) {
			unsigned int cachedPoint = j * COMPONENTS_2D;
			// Don't calculate the forces of a point on itself
			if(tileStart + cachedPoint == pointId) {
				continue;
			}

			float2 otherPoint = (float2) (tilePoints[cachedPoint + 0], tilePoints[cachedPoint + 1]);

			posDelta += (calculatePointForce(myPos, otherPoint));
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Calculate force from walls


	myPos += posDelta;// * timeDelta;

	// Clamp myPos to be within the walls
	myPos.x = clamp(myPos.x, walls[0], walls[1]);
	myPos.y = clamp(myPos.y, walls[2], walls[3]);

	outputPositions[pointId + 0] = myPos.x;
	outputPositions[pointId + 1] = myPos.y;

	return;
}


float2 calculatePointForce(float2 a, float2 b) {
	float r = (b.x - a.x)*(b.x - a.x) + (b.y - a.y)*(b.y - a.y); //distance(a, b);

	if(r < FLT_EPSILON) {
		return (float2) (0.0f, 0.0f);
	}

	float2 force = ((float2) ((b.x - a.x)/r, (b.y - a.y)/r)) * POINT_REPULSION * -1.0f;

	// TODO: Clamp the force at some maximum
	return force;
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


