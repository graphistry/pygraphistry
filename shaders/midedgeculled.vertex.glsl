precision mediump float;

#define W_VAL 1.0
#define Z_VAL 0.0
#define SENSITIVITY 0.99
#define M -33.33 // 1.0 / (1.02 - 1.05);
#define B 35.0

uniform mat4 mvp;
attribute vec2 curPos;
attribute vec2 startPos;
attribute vec2 endPos;

attribute vec4 edgeColor;
varying vec4 eColor;

uniform float edgeOpacity;
uniform float isOpaque; //map non-transparent to 1.0

void main(void) {
    vec4 pos = mvp * vec4(curPos.x, 1.0 * curPos.y, Z_VAL, W_VAL);

    vec4 sPos = mvp * vec4(startPos.x, startPos.y, Z_VAL, W_VAL);
    float furthestS = max(abs(sPos.x), abs(sPos.y));
    float remappedS = clamp(M * furthestS + B, 0.0, 1.0);

    vec4 ePos = mvp * vec4(endPos.x, endPos.y, Z_VAL, W_VAL);
    float furthestE = max(abs(ePos.x), abs(ePos.y));
    float remappedE = clamp(M * furthestE + B, 0.0, 1.0);

    float alpha = dot(pow(vec2(remappedS, remappedE), vec2(4.0, 4.0)), vec2(0.5, 0.5));

    pos.z = 1.0 - alpha;
    eColor = isOpaque > 0.5 ? (alpha > 0.10 ? vec4(edgeColor.g, edgeColor.b, edgeColor.a, 1.0) : vec4(0.0,0.0,0.0,0.0)) : vec4(edgeColor.xyz, edgeOpacity * alpha);
    gl_Position = pos;
}
