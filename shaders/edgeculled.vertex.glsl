precision mediump float;

#define W_VAL 1.0
#define Z_VAL 0.0

uniform mat4 mvp;

attribute vec2 curPos;
attribute vec4 edgeColor;

varying vec4 eColor;

uniform float edgeOpacity;

void main(void) {
    vec4 pos = mvp * vec4(curPos.x, curPos.y, Z_VAL, W_VAL);

    float furthestComponent = max(abs(pos.x), abs(pos.y));
    float m = 1.0 / (1.02 - 1.05);
    float b = -1.0 * m * 1.05;
    float remapped = m * furthestComponent + b;
    float alpha = clamp(remapped, 0.0, 1.0);

    pos.z = 1.0 - alpha;
    gl_Position = pos;
    eColor =  vec4(edgeColor.xyz, edgeOpacity * alpha);
}
