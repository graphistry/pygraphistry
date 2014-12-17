#ifdef GL_ES
precision highp float;
#endif

varying vec4 eColor;
varying float alpha;

void main(void) {
    gl_FragColor = vec4(eColor.x, eColor.y, eColor.z, alpha);
}
