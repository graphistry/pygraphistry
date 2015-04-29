#ifdef GL_ES
precision highp float;
#endif

varying vec4 eColor;

void main(void) {
    gl_FragColor = eColor;
}
