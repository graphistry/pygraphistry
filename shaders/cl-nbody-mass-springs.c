// Number of elements per 2d point
#define COMPONENTS_2D 2

#define POINT_REPULSION 1.0f
// #define EDGE_REPULSION 0.5f

// #define SPRING_LENGTH 0.1f
// #define SPRING_FORCE 0.1f

// TODO: Add in a repulsive force from the four walls
// TODO: Allow the wall coordinates to be passed as a kernel argument

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

	float4 walls = (float4) (0.05f, 0.95f, 0.05f, 0.95f);

	unsigned int threadLocalId = (unsigned int) get_local_id(0);
	unsigned int pointId = (unsigned int) get_global_id(0) * COMPONENTS_2D;

	// // The point we're updating
	// // TODO: Convert to vector read
	float2 myPos = (float2) (inputPositions[pointId + 0], inputPositions[pointId + 1]);

	// Points per tile = threads per workgroup
	unsigned int tileSize = (unsigned int) get_local_size(0);
	unsigned int numTiles = numPoints / tileSize;
	numTiles = max(numTiles, (uint) 1);

	float2 posDelta = (float2) (0.0f, 0.0f);

	for(unsigned int tile = 0; tile < numTiles; tile++) {
		unsigned int tileStart = (tile * (tileSize * COMPONENTS_2D));
		unsigned int tilePointId = tileStart + (threadLocalId * COMPONENTS_2D);

		// TODO: Convert to a bulk local, synchronized read
		tilePoints[(COMPONENTS_2D * threadLocalId) + 0] = inputPositions[tilePointId + 0];
		tilePoints[(COMPONENTS_2D * threadLocalId) + 1] = inputPositions[tilePointId + 1];

		barrier(CLK_LOCAL_MEM_FENCE);

		for(unsigned int j = 0; j < tileSize; j++) {
			unsigned int cachedPoint = j * COMPONENTS_2D;
			// Skip calculating forces if the other point is really this point (that is, don't
			// calculate the forces of a point on itself.)
			if(tileStart + cachedPoint == pointId) {
				continue;
			}

			float2 otherPoint = (float2) (tilePoints[cachedPoint + 0], tilePoints[cachedPoint + 1]);

			// Calculate force as POINT_REPULSION * (1/(distance^2))
			float dist = distance(myPos, otherPoint);
			float2 dir;
			float force;

			if(dist > 0) {
				// Force magnitude is POINT_REPULSION * (1/(distance^2))
				force = POINT_REPULSION * (1.0f/(dist*dist));
				// Force direction is the direction of the other point
				dir = otherPoint - myPos;
			} else {
				// If dist <= 0, leave force as 100% of FORCE_REPULSION
				force = POINT_REPULSION;
				// Force direction is set to a (sorta) random direction
				dir = (float2) (cachedPoint - pointId, pointId - cachedPoint);
			}

			force = clamp(force, 0.0f, POINT_REPULSION);
			dir = normalize(dir);

			// float2 change =
			posDelta += dir * force * timeDelta * -1;;
		}

		// Calculate force from walls

		barrier(CLK_LOCAL_MEM_FENCE);
	}


	myPos += posDelta;

	// Clamp myPos to be within the walls
	myPos.x = clamp(myPos.x, walls[0], walls[1]);
	myPos.y = clamp(myPos.y, walls[2], walls[3]);

	// TODO: Convert to vector write
	outputPositions[pointId + 0] = myPos.x;
	outputPositions[pointId + 1] = myPos.y;
	// outputPositions[pointId + 2] = myPos.z;
	// outputPositions[pointId + 3] = myPos.w;

	return;
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
