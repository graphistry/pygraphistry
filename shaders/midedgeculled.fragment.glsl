precision mediump float;

varying vec4 eColor;
uniform float isOpaque; //map non-transparent to 1.0

void main(void) {

    if (isOpaque > 0.5) {
        gl_FragColor = vec4(eColor.xyz, 1.0);
    } else {
        gl_FragColor = eColor;
    }

}
