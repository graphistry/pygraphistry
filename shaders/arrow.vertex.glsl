precision highp float;

#define W_VAL 1.0
#define Z_VAL 0.0

uniform mat4 mvp;
uniform float zoomScalingFactor;
uniform float maxPointSize;
uniform float screenWidth;
uniform float screenHeight;

attribute vec2 curPos;
attribute float pointSize;
attribute vec4 arrowColor;
attribute vec2 edgeVec;

varying vec4 aColor;

void main(void) {
    float radius = clamp(zoomScalingFactor * pointSize, 7.0, maxPointSize) / 2.0;
    vec4 offset = mvp * vec4(edgeVec.xy, Z_VAL, 0.0);
    vec4 noffset = normalize(offset);
    
    vec4 pos0 = mvp * vec4(curPos.xy, Z_VAL, W_VAL);
    //vec4 pos = pos0;

    //pos.x = pos0.x + (radius / screenWidth) * noffset.x;
    //pos.y = pos0.y + (radius / screenHeight) * noffset.y;
    //
    vec4 pos = pos0 + noffset / 100.0;

    float furthestComponent = max(abs(pos0.x), abs(pos0.y));
    float m = 1.0 / (1.02 - 1.05);
    float b = -1.0 * m * 1.05;
    float remapped = m * furthestComponent + b;
    float alpha = clamp(remapped, 0.0, 1.0);

    pos.z = 1.0 - alpha;
    gl_Position = pos;
    aColor = vec4(arrowColor.xyz, alpha);
}
