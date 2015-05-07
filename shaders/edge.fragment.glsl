#ifdef GL_ES
precision highp float;
#endif

varying vec4 eColor;
varying vec3 eColorFullOpacity;

void main(void) {
    if (eColor.w <= 0.0) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
    } else {
        gl_FragColor = vec4(eColorFullOpacity.x, eColorFullOpacity.y,
                eColorFullOpacity.z, 1.0);
    }
}
