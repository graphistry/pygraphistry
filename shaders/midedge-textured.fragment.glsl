precision highp float;

uniform sampler2D uSampler;

varying vec2 vColorCoord;

void main(void) {
	vec2 colorXY = vec2(clamp((vColorCoord.x + 1.0)/2.0, 0.0, 1.0), clamp((vColorCoord.y + 1.0)/2.0, 0.0, 1.0));
	vec4 texColor = texture2D(uSampler, colorXY);
	texColor.a = 0.2;
	gl_FragColor = texColor;
}