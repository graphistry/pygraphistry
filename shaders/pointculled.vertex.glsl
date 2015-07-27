precision highp float;

#define W_VAL 1.0
#define Z_VAL 0.1

attribute vec2 curPos;
attribute float pointSize;
attribute vec4 pointColor;

varying vec4 vColor;

uniform mat4 mvp;
uniform float fog;
uniform float stroke;
uniform float zoomScalingFactor;
uniform float maxPointSize;
uniform float minPointSize;
uniform float pointOpacity;

void main(void) {
    if (stroke > 0.0) {
        gl_PointSize = clamp(zoomScalingFactor * pointSize, minPointSize, maxPointSize);
    } else {
        gl_PointSize = stroke + clamp(zoomScalingFactor * pointSize, minPointSize, maxPointSize);
    }

    vec4 pos = mvp * vec4(curPos.xy, Z_VAL, W_VAL);
    gl_Position = pos;

    vColor = vec4(stroke > 0.0 ? 0.5 * pointColor.xyz : pointColor.xyz, pointOpacity);
}
