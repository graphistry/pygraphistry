precision mediump float;

#define W_VAL 1.0
#define Z_VAL 0.0
#define SENSITIVITY 0.99
#define M -33.33 // 1.0 / (1.02 - 1.05);
#define B 35.0

uniform mat4 mvp;
attribute vec2 curPos;

attribute vec4 edgeColor;
varying vec4 eColor;

void main(void) {
    vec4 pos = mvp * vec4(curPos.x, 1.0 * curPos.y, Z_VAL, W_VAL);
    float furthestComponent = max(abs(pos.x), abs(pos.y));
    float remapped = M * furthestComponent + B;

    float alpha = clamp(remapped, 0.0, 0.5);
    pos.z = 1.0 - alpha;

    eColor = vec4(edgeColor.xyz, alpha);
    gl_Position = pos;
}
