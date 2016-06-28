precision highp float;

#define W_VAL 1.0
#define Z_VAL 0.0

uniform mat4 mvp;
// uniform vec2 uResolution;

attribute vec2 curPos;
attribute vec2 aColorCoord;

varying vec2 vColorCoord;

void main(void) {
	vec4 colorLoc = vec4(aColorCoord.x, aColorCoord.y, Z_VAL, W_VAL);
	vColorCoord = colorLoc.xy;

    gl_Position = mvp * vec4(curPos.x, curPos.y, Z_VAL, W_VAL);
}
