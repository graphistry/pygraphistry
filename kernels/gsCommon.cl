// The length of the 'randValues' array
#define RAND_LENGTH 73 //146


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
