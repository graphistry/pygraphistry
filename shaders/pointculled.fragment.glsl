precision highp float;

varying vec4 vColor;
uniform float fog;

void main(void) {

    float radius = length(gl_PointCoord - 0.5);
    float alpha = vColor.w * step(radius, 0.5);
    vec4 fillColor = vec4(vColor.x, vColor.y, vColor.z, alpha);
    gl_FragColor = fog > 1.0 ? fillColor : vColor;
}
