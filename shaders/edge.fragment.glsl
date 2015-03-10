#ifdef GL_ES
precision highp float;
#endif

varying vec4 eColor;
varying float alpha;

void main(void) {
    if (alpha <= 0.0) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
    } else {
        gl_FragColor = vec4(eColor.x, eColor.y, eColor.z, 1);
    }
}
