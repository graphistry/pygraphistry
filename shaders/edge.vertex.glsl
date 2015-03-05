precision mediump float;

#define W_VAL 1.0
#define Z_VAL 0.0

uniform mat4 mvp;
attribute vec2 curPos;

attribute vec4 edgeColor;
varying vec4 eColor;

void main(void) {
    vec4 pos = mvp * vec4(curPos.x, curPos.y, Z_VAL, W_VAL);
    gl_Position = pos;
    eColor = edgeColor;
}
