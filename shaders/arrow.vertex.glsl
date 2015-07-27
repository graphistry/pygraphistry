precision highp float;

#define W_VAL 1.0
#define Z_VAL 0.0

uniform mat4 mvp;
uniform float zoomScalingFactor;
uniform float maxPointSize;
uniform float maxScreenSize;
uniform float maxCanvasSize;
uniform float edgeOpacity;

attribute vec2 startPos;     // Start position of edge
attribute vec2 endPos;       // End position of edge
attribute float normalDir;   // Direction of normal vector.
                             // 0 for tip vertex Wing vertices -> 1.0/-1.0
attribute float pointSize;
attribute vec4 arrowColor;

varying vec4 aColor;

void culledAlpha(in vec2 pos, out float alpha) {
    float furthestComponent = max(abs(pos.x), abs(pos.y));
    float m = 1.0 / (1.02 - 1.05);
    float b = -1.0 * m * 1.05;
    float remapped = m * furthestComponent + b;
    alpha = clamp(remapped, 0.0, 1.0);
}

void main(void) {
    vec2 arrow = startPos - endPos;
    vec4 edgeVec = vec4(arrow.xy, Z_VAL, 0.0);
    vec4 edgeVecN = normalize(edgeVec);
    vec4 normalVecN = vec4(-edgeVecN.y, edgeVecN.x, edgeVecN.z, edgeVecN.w);
    vec4 screenEdgeVec = mvp * edgeVec;
    float edgeLength = length(screenEdgeVec);

    vec4 pointPos = vec4(endPos.xy, Z_VAL, W_VAL);

    vec4 offset = abs(normalDir) * edgeVecN + 0.5 * normalDir * normalVecN;
    float semanticSizeFactor = clamp(-75.0 * edgeLength + 135.0, 60.0, 120.0);
    //float semanticSizeFactor = clamp(-100.0 * edgeLength + 170.0, 70.0, 160.0);
    float arrowSize = 0.8 * maxScreenSize / semanticSizeFactor;
    vec4 pos0 = pointPos + arrowSize * offset;

    // Displace arrow to move the tip from the center to the edge of the point.
    float pointRadius = clamp(zoomScalingFactor * pointSize, 7.0, maxPointSize);
    vec4 displacement = normalize(screenEdgeVec) * pointRadius / maxCanvasSize;
    vec4 pos = mvp * pos0 + displacement;

    // Cull arrow if its point is offscreen.
    vec4 screenPointPos = mvp * pointPos;
    float alpha;
    culledAlpha(screenPointPos.xy, alpha);

    // Clip small arrows by placing them behing the far plane.
    pos.z = edgeLength < 0.05 ? 100.0 : 1.0 - alpha;

    gl_Position = pos;
    aColor = vec4(arrowColor.xyz, edgeOpacity * alpha);
}
