precision mediump float;

#define W_VAL 1.0
#define Z_VAL 0.0

uniform mat4 mvp;
attribute vec2 curPos;

attribute float pointSize;

attribute vec4 pointColor;
varying vec4 vColor;

attribute float isHighlighted;

void main(void) {
    gl_PointSize = isHighlighted > 0.0 ? isHighlighted : clamp(pointSize, 0.125, 50.0);

    vec4 pos = vec4(curPos.x, curPos.y, Z_VAL, W_VAL);
    gl_Position = mvp * pos;

    vColor = pointColor;
}
