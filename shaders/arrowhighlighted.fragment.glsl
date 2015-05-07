#ifdef GL_ES
precision highp float;
#endif

varying vec4 aColor;

void main(void) {
    gl_FragColor = aColor;
}
