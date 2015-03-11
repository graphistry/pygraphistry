precision mediump float;

#define W_VAL 1.0
#define Z_VAL 0.0
#define SENSITIVITY 0.99

uniform mat4 mvp;
attribute vec2 curPos;
varying float alpha;

attribute vec4 edgeColor;
varying vec4 eColor;

void main(void) {
    vec4 pos = mvp * vec4(curPos.x, 1.0 * curPos.y, Z_VAL, W_VAL);

    float furthestComponent = max(abs(pos.x), abs(pos.y));
    float m = 1.0 / (1.02 - 1.05);
    float b = -1.0 * m * 1.05;
    float remapped = m * furthestComponent + b;

    alpha = clamp(remapped, 0.0, 1.0);
    gl_Position = pos;
    eColor = edgeColor;
}
