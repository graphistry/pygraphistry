precision mediump float;
uniform mat3 mvp;
attribute vec2 curPos;

void main(void) {
    float w = 1.0;

    vec3 pos = mvp * vec3(curPos[0], curPos[1], w);
    gl_Position = vec4(pos[0], pos[1], 0.0, pos[2]);
}
