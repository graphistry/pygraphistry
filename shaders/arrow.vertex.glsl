precision mediump float;

#define W_VAL 1.0
#define Z_VAL 0.0

uniform mat4 mvp;
uniform float zoomScalingFactor;
uniform float maxPointSize;

attribute vec2 curPos;
attribute float pointSize;
attribute vec4 arrowColor;
attribute vec2 edgeVec;

varying vec4 aColor;

void main(void) {
    float radius = clamp(zoomScalingFactor * pointSize, 7.0, maxPointSize) / 2000.0;
    vec4 offset = mvp * vec4(edgeVec.xy, Z_VAL, 0.0);
    
    vec4 pos0 = mvp * vec4(curPos.xy, Z_VAL, W_VAL);
    vec4 pos = pos0 + radius * normalize(offset);

    float furthestComponent = max(abs(pos0.x), abs(pos0.y));
    float m = 1.0 / (1.02 - 1.05);
    float b = -1.0 * m * 1.05;
    float remapped = m * furthestComponent + b;
    float alpha = clamp(remapped, 0.0, 1.0);

    pos.z = 1.0 - alpha;
    gl_Position = pos;
    aColor = vec4(arrowColor.xyz, alpha);
}
