#ifdef GL_ES
precision highp float;
#endif

varying vec4 eColor;
varying float alpha;

void main(void) {
    gl_FragColor = vec4(eColor.y, eColor.z, eColor.w, alpha < 0.0 ? 0.01 : (0.2 * clamp(alpha, 0.1, 0.2)));
}
