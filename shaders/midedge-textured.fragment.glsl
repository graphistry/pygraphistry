precision mediump float;

uniform sampler2D uSampler;

void main(void) {
	vec4 texColor = texture2D(uSampler, vec2(gl_FragCoord.x/700.0, gl_FragCoord.y/700.0));
	texColor.a = 0.1;
	gl_FragColor = texColor;

    // gl_FragColor = vec4(0.5, 0.5, 0.5, 0.1);
}