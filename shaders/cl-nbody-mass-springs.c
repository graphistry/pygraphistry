// Number of elements per 2d point
#define COMPONENTS_2D 2

#define POINT_REPULSION 0.00001
#define EDGE_REPULSION 0.5

#define SPRING_LENGTH 0.1
#define SPRING_FORCE 0.1

__kernel void nbody_compute_repulsion(
	unsigned int numPoints,
	__global float* inputPositions,
	__global float* outputPositions,
	float timeDelta,
	__local float* tilePoints)
{
	// use async_work_group_copy() and wait_group_events() to fetch the data from global to local
	// use vloadn() and vstoren() to read/write vectors.
	// use clamp() to ensure that points are within (-1,1)
	// Effects of points should generally be proportional to 1/(distance^2)

	float4 walls = (float4) (0.05, 0.95, 0.05, 0.95);

	unsigned int threadLocalId = get_local_id(0);
	unsigned int pointId = get_global_id(0) * COMPONENTS_2D;

	// // The point we're updating
	// // TODO: Convert to vector read
	float2 myPos = (float2) (inputPositions[pointId + 0], inputPositions[pointId + 1]);

	// Points per tile = threads per workgroup
	unsigned int tileSize = get_local_size(0);
	unsigned int numTiles = numPoints / tileSize;

	float2 posDelta = (float2) (0.0f, 0.0f);

	for(int tile = 0; tile < numTiles; tile++) {
		unsigned int tilePointId = (tile * tileSize) + threadLocalId;

		// TODO: Convert to a bulk local, synchronized read
		tilePoints[(COMPONENTS_2D * threadLocalId) + 0] = inputPositions[(COMPONENTS_2D * tilePointId) + 0];
		tilePoints[(COMPONENTS_2D * threadLocalId) + 1] = inputPositions[(COMPONENTS_2D * tilePointId) + 1];

		barrier(CLK_LOCAL_MEM_FENCE);

		for(unsigned int j = 0; j < tileSize; j++) {
			float2 otherPoint = (float2) (tilePoints[(COMPONENTS_2D * j) + 0], tilePoints[(COMPONENTS_2D * j) + 1]);

			float2 dir = otherPoint - myPos;
			dir = normalize(dir);

			float dist = distance(myPos, otherPoint);
			float force = POINT_REPULSION * ((1.0/(dist*dist))/1000);

			float2 change = dir * force * timeDelta;
			posDelta += change;
		}

		myPos += posDelta;

		// Clamp myPos to be within the walls
		myPos.x = clamp(myPos.x, walls[0], walls[1]);
		myPos.y = clamp(myPos.y, walls[2], walls[3]);

		// Calculate force from walls

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// TODO: Convert to vector write
	outputPositions[pointId + 0] = myPos.x;
	outputPositions[pointId + 1] = myPos.y;
	// outputPositions[pointId + 2] = myPos.z;
	// outputPositions[pointId + 3] = myPos.w;

	return;
}


__kernel void nbody2d_compute_springs(
	unsigned int numEdges,
	__global unsigned int* springList,
	__global float* springPositions,
	__global float* inputPositions,
	__global float* outputPositions,
	float timeDelta)
{
	// From Hooke's Law, we generally have that the force exerted by a spring is given by
	//	F = -k * X, where X is the distance the spring has been displaced from it's natural
	// distance, and k is some constant positive real number.
	return;
}