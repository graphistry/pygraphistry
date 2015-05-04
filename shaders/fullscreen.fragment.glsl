#ifdef GL_ES
precision highp float;
#endif

uniform sampler2D uSampler;
varying vec2 textureCoords;

void main(void) {
    // gl_FragColor = texture2D(uSampler, textureCoords);
    gl_FragColor = vec4(0.69, 0.269, 0.145, 1);
}
