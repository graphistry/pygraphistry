precision mediump float;
uniform mat4 mvp;
attribute vec4 curPos;
attribute vec4 curVel;
varying vec4 vColor;

void main(void) {
    vec4 pos;
    pos.xyz = curPos.xyz;
    pos.w  = 1.0;
    gl_Position = mvp * pos;

    float maxSize = 8.0;
    float size = maxSize * (1.0 - curPos.z);
    if(size < 1.0) size = 1.0;
    if(size > maxSize) size = maxSize;

    float vel = sqrt(curVel.x*curVel.x + curVel.y*curVel.y + curVel.z*curVel.z);
    float r = abs(curVel.x)/vel;
    float g = abs(curVel.y)/vel;
    float b = abs(curVel.z)/vel;

    vColor = vec4(r, g, b, 0.8);

    gl_PointSize  = size;
}