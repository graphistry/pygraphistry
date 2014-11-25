#ifdef GL_ES
precision highp float;
#endif

varying float alpha;

void main(void) {
    gl_FragColor = vec4(0.5, 0.5, 0.5, 0.1 * clamp(alpha, 0.0, 1.0));
}