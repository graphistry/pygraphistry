// Speed tuning parameters
#define KS    0.1f
#define KSMAX 10.0f

#define REPULSION_OVERLAP 0.00000001f
#define DEFAULT_NODE_SIZE 0.000001f
#define EPSILON 0.00001f // bound whether d(a,b) == 0

#define IS_DISSUADE_HUBS(flags)   (flags & 1)
#define IS_LIN_LOG(flags)         (flags & 2)

//#define NOGRAVITY
//#define NOREPULSION
//#define NOATTRACTION
