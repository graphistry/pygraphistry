precision highp float;

#define W_VAL 1.0
#define Z_VAL 0.0
#define SENSITIVITY 0.5

uniform mat4 mvp;
attribute vec2 curPos;

attribute float pointSize;

attribute vec4 pointColor;
varying vec4 vColor;

uniform float fog;
uniform float stroke;
uniform float zoomScalingFactor;

void main(void) {
    if (stroke > 0.0) {
        gl_PointSize = clamp(zoomScalingFactor * pointSize, 5.0, 50.0);
    } else {
        gl_PointSize = stroke + clamp(zoomScalingFactor * pointSize, 5.0, 50.0);
    }

    vec4 pos = mvp * vec4(curPos.x, 1.0 * curPos.y, Z_VAL, W_VAL);
    gl_Position = pos;

    if (fog > 1.0) {
        float furthestComponent = max(abs(pos.x), abs(pos.y));
        float remapped = (-furthestComponent + SENSITIVITY) / SENSITIVITY;
        float alpha =  remapped < 0.0 ? 0.6 : (0.6 + clamp(remapped, 0.0, 0.2));
        vColor = vec4(pointColor.x, pointColor.y, pointColor.z, alpha);
    } else {
        vColor = vec4(pointColor.xyz, 1.0);
    }
}
