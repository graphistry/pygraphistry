// Speed tuning parameters
#define KS    0.1f
#define KSMAX 10.0f

#define REPULSION_OVERLAP 0.00000001f
#define DEFAULT_NODE_SIZE 0.000001f
#define EPSILON 0.00001f // bound whether d(a,b) == 0

#define IS_PREVENT_OVERLAP(flags) (flags & 1)
#define IS_STRONG_GRAVITY(flags)  (flags & 2)
#define IS_DISSUADE_HUBS(flags)   (flags & 4)
#define IS_LIN_LOG(flags)         (flags & 8)

//#define NOGRAVITY
//#define NOREPULSION
//#define NOATTRACTION

float attractionForce(const float2 distVec, const float n1Size, const float n2Size,
                      const uint n1Degree, const float weight, const bool preventOverlap,
                      const uint edgeInfluence, const bool linLog, const bool dissuadeHubs) {

    const float weightMultiplier = edgeInfluence == 0 ? 1.0f
                                 : edgeInfluence == 1 ? weight
                                                      : pown(weight, edgeInfluence);

    const float dOffset = preventOverlap ? n1Size + n2Size : 0.0f;
    const float dist = length(distVec) - dOffset;

    float aForce;
    if (preventOverlap && dist < EPSILON) {
        aForce = 0.0f;
    } else {
        const float distFactor = (linLog ? log(1.0f + dist) : dist);
        const float n1Deg = (dissuadeHubs ? n1Degree + 1.0f : 1.0f);
        aForce = weightMultiplier * distFactor / n1Deg;
    }

#ifndef NOATTRACTION
    return aForce;
#else
    return 0.0f;
#endif
}
