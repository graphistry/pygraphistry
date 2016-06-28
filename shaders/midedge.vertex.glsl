precision mediump float;

#define W_VAL 1.0
#define Z_VAL 0.0

uniform mat4 mvp;
attribute vec2 curPos;
attribute vec4 edgeColor;
attribute vec4 eColor;

void main(void) {
    vec4 pos = vec4(curPos.x, 1.0 * curPos.y, Z_VAL, W_VAL);
    gl_Position = mvp * pos;
}
