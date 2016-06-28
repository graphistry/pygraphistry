precision mediump float;

#define W_VAL 1.0
#define Z_VAL 0.0

uniform mat4 mvp;
attribute vec2 curPos;

attribute vec4 edgeColor;
varying vec4 eColor;
varying vec3 eColorFullOpacity;

void main(void) {
    vec4 pos = mvp * vec4(curPos.x, curPos.y, Z_VAL, W_VAL);

    // Calculate whether or not it should be shown as a boolean value
    float furthestComponent = max(abs(pos.x), abs(pos.y));
    float m = 1.0 / (1.02 - 1.05);
    float b = -1.0 * m * 1.05;
    float remapped = m * furthestComponent + b;
    float alpha = clamp(remapped, 0.0, 1.0);

    gl_Position = pos;

    // eColor = edgeColor;
    eColor = vec4(edgeColor.g, edgeColor.b, edgeColor.a, alpha);
    eColorFullOpacity = vec3(edgeColor.g, edgeColor.b, edgeColor.a);
}
