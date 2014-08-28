#ifdef GL_ES
precision highp float;
#endif

varying vec4 vColor;

void main(void) {
    gl_FragColor = vec4(vColor.y, vColor.z, vColor.w, vColor.x * step(length(gl_PointCoord - 0.5), 0.5));
}
