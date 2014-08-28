precision mediump float;
uniform mat3 mvp;
attribute vec2 curPos;

attribute float pointSize;

attribute vec4 pointColor;
varying vec4 vColor;

void main(void) {
    float w = 1.0;

    gl_PointSize = clamp(pointSize, 0.125, 10.0);

    vec3 pos = mvp * vec3(curPos[0], curPos[1], w);
    gl_Position = vec4(pos[0], pos[1], 0.0, pos[2]);

    vColor = pointColor;
}
