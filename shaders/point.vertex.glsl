precision mediump float;

#define W_VAL 1.0
#define Z_VAL 0.0

uniform mat4 mvp;
attribute vec2 curPos;

attribute float pointSize;

attribute vec4 pointColor;
varying vec4 vColor;

uniform float zoomScalingFactor;

void main(void) {
    gl_PointSize = clamp(zoomScalingFactor * pointSize, 5.0, 50.0);

    vec4 pos = vec4(curPos.x, curPos.y, Z_VAL, W_VAL);
    gl_Position = mvp * pos;

    vColor = vec4(pointColor.g, pointColor.b, pointColor.a, 1.0);
}
