#ifdef GL_ES
precision highp float;
#endif

uniform sampler2D uSampler;
varying vec2 textureCoords;

void main(void) {
    gl_FragColor = texture2D(uSampler, textureCoords);
}
