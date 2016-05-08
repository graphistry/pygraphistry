#define REPULSION_OVERLAP 0.00000001f
#define DEFAULT_NODE_SIZE 0.000001f
#define EPSILON 0.00001f // bound whether d(a,b) == 0

#define IS_STRONG_GRAVITY(flags)  (flags & 2)
#define IS_DISSUADE_HUBS(flags)   (flags & 4)
#define IS_LIN_LOG(flags)         (flags & 8)

inline float repulsionForce(const float distSquared, const uint n1DegreePlusOne, const uint n2DegreePlusOne,
                            const float scalingRatio) {
    const int degreeProd = (n1DegreePlusOne * n2DegreePlusOne);
    float force;

    // We use dist squared instead of dist because we want to normalize the
    // distance vector as well
    force = scalingRatio * degreeProd / distSquared;

#ifndef NOREPULSION
    // Assuming always positive.
    // return clamp(force, 0.0f, 1000000.0f);
    return min(force, 1000000.0f);
#else
    return 0.0f;
#endif
}

float attractionForce(const float2 distVec, const float n1Size, const float n2Size,
                      const uint n1Degree, const float weight,
                      const uint edgeInfluence, const bool linLog, const bool dissuadeHubs) {

    const float weightMultiplier = edgeInfluence == 0 ? 1.0f
                                 : edgeInfluence == 1 ? weight
                                                      : pown(weight, edgeInfluence);
    debug3("Weight: %f    Mult:%f\n", weight, weightMultiplier);
    const float dOffset = 0.0f;
    const float dist = length(distVec) - dOffset;

    float aForce;
    const float distFactor = (linLog ? log(1.0f + dist) : dist);
    const float n1Deg = (dissuadeHubs ? n1Degree + 1.0f : 1.0f);
    aForce = weightMultiplier * distFactor / n1Deg;

#ifndef NOATTRACTION
    return aForce;
#else
    return 0.0f;
#endif
}

inline float gravityForce(const float gravity, const uint n1Degree, const float2 centerVec,
                          const bool strong) {

    float gForce = gravity * (n1Degree + 1.0f);
    if (strong) {
        gForce *= fast_length(centerVec);
    }

#ifndef NOGRAVITY
    return gForce;
#else
    return 0.0f;
#endif
}

