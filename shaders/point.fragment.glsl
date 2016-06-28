#ifdef GL_ES
precision highp float;
#endif

varying vec4 vColor;

void main(void) {
    gl_FragColor = vec4(vColor.x, vColor.y, vColor.z, vColor.w * step(length(gl_PointCoord - 0.5), 0.5));
}
