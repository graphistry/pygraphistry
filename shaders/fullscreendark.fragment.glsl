#ifdef GL_ES
precision highp float;
#endif

uniform sampler2D uSampler;
varying vec2 textureCoords;

void main(void) {
    // TODO: Do this computation once before writing the texture.
    vec4 color = texture2D(uSampler, textureCoords);
    gl_FragColor = vec4(color.r, color.g, color.b, 0.25);
}
