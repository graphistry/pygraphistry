// Number of elements per 2d point
#define COMPONENTS_2D 2

#define POINT_REPULSION 1.0
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
	
	unsigned int localId = get_local_id(0);
	unsigned int pointId = get_global_id(0) * COMPONENTS_2D;
	
	// // The point we're updating
	// // TODO: Convert to vector read
	// float4 myPos = (float4) (inputPositions[pointId + 0], inputPositions[pointId + 1], inputPositions[pointId + 2], inputPositions[pointId + 3]);
	float2 myPos = (float2) (inputPositions[pointId + 0], inputPositions[pointId + 1]);
	
	// Points per tile = threads per workgroup
	// unsigned int tileSize = get_local_size(0);
	// unsigned int numTiles = numPoints / tileSize;
	
	// for(int i = 0; i < numTilesl i++) {
	// 	unsigned int tilePointId = (i * tileSize) + localId;
		
	// 	// TODO: Convert to a bulk local, synchronized read
	// 	tilePoints[(4*localId) + 0] = inputPositions
	// }
	
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