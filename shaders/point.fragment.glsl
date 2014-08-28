#ifdef GL_ES
precision highp float;
#endif

void main(void) {
    gl_FragColor = vec4(1.0, 0.3, 0.3, step(length(gl_PointCoord - 0.5), 0.5));
}
