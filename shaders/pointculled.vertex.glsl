precision highp float;

#define W_VAL 1.0
#define Z_VAL 0.0
#define SENSITIVITY 0.95

uniform mat4 mvp;
attribute vec2 curPos;

attribute float pointSize;

attribute vec4 pointColor;
varying vec4 vColor;

attribute float isHighlighted;
uniform float fog;

void main(void) {
    gl_PointSize = isHighlighted > 0.0 ? isHighlighted : clamp(pointSize, 0.125, 10.0);

    vec4 pos = mvp * vec4(curPos.x, 1.0 * curPos.y, Z_VAL, W_VAL);
    gl_Position = pos;

    if (fog > 1.0) {
        float furthestComponent = max(abs(pos.x), abs(pos.y));
        float remapped = (-furthestComponent + SENSITIVITY) / SENSITIVITY;
        float alpha = remapped < 0.0 ? -20.0 : clamp(remapped, 0.0, 1.0);
        vColor = vec4(pointColor.y, pointColor.z, pointColor.w, alpha);
    } else {
        vColor = vec4(pointColor.xyz, 1.0);
    }
}