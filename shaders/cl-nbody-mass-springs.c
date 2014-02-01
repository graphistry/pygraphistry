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
	
	unsigned int tid = get_local_id(0);
	unsigned int gid = get_global_id(0);
	
	// Points per tile = threads per workgroup
	unsigned int tileSize = get_local_size(0);
	unsigned int numTiles = numPoints / tileSize;
	
	// The point we're updating
	float4 myPos = (float4) (inputPositions[4*gid + 0], inputPositions[4*gid + 1], inputPositions[4*gid + 2], inputPositions[4*gid + 3]);
	
	outputPositions[4*gid + 0] = myPos.x;
	outputPositions[4*gid + 1] = myPos.y;
	outputPositions[4*gid + 2] = myPos.z;
	outputPositions[4*gid + 3] = myPos.w;
	
	return;
}


__kernel void nbody_compute_springs(
	unsigned int numEdges,
	__global unsigned int* edges,
	__global float* inputPositions,
	__global float* outputPositions,
	float timeDelta) 
{
	// From Hooke's Law, we generally have that the force exerted by a spring is given by
	//	F = -k * X, where X is the distance the spring has been displaced from it's natural 
	// distance, and k is some constant positive real number.
	return;
}