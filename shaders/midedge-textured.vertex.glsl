#define W_VALUE 1.0

precision mediump float;

uniform mat3 mvp;
// uniform vec2 uResolution;

attribute vec2 curPos;
attribute vec2 aColorCoord;

varying vec2 vColorCoord;

void main(void) {
	vec3 colorLoc = (mvp * vec3(aColorCoord.x, aColorCoord.y, W_VALUE));
	colorLoc.x = colorLoc.x / colorLoc.z;
	colorLoc.y = colorLoc.y / colorLoc.z;
	vColorCoord = colorLoc.xy;

    vec3 pos = mvp * vec3(curPos[0], curPos[1], W_VALUE);
    gl_Position = vec4(pos[0], pos[1], 0.0, pos[2]);
}
