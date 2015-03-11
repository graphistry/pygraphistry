#include "common.h"
#include "barnesHut/barnesHutCommon.h"

__kernel void move_bodies(
        float scalingRatio, float gravity, unsigned int edgeWeightInfluence, unsigned int flags,
        __global volatile float *x_cords,
        __global float *y_cords,
        __global float *accx,
        __global float * accy,
        __global volatile int* children,
        __global float* mass,
        __global int* start,
        __global int* sort,
        __global float* global_x_mins,
        __global float* global_x_maxs,
        __global float* global_y_mins,
        __global float* global_y_maxs,
        __global float* swings,
        __global float* tractions,
        __global int* count,
        __global volatile int* blocked,
        __global volatile int* step,
        __global volatile int* bottom,
        __global volatile int* maxdepth,
        __global volatile float* radiusd,
        __global volatile float* globalSpeed,
        unsigned int step_number,
        float width,
        float height,
        const int num_bodies,
        const int num_nodes,
        __global float2* pointForces,
        float tau
){

    /*const float dtime = 0.025f;*/
    /*const float dthf = dtime * 0.5f;*/
    float velx, vely;

    /*printf("Gravity: %f\n", gravity);*/
    int inc = get_global_size(0);
    for (int i = get_global_id(0); i < num_bodies; i+= inc) {
        /*velx = accx[i] * dthf;*/
        /*vely = accy[i] * dthf;*/
        float center_distance_x = width/2 - x_cords[i];
        float center_distance_y = height/2 - y_cords[i];
        float gravity_force = gravity / sqrt(center_distance_x*center_distance_x + center_distance_y * center_distance_y);
        velx = (accx[i] * 0.00001f) + (0.01f * gravity_force * center_distance_x);
        vely = (accy[i] * 0.00001f) + (0.01f * gravity_force * center_distance_y);

        x_cords[i] += velx;
        y_cords[i] += vely;
    }
}
