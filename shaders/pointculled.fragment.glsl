precision highp float;

varying vec4 vColor;
uniform float fog;
uniform float stroke;

void main(void) {

    float radius = length(gl_PointCoord * 2.0 - vec2(1.0));
    float alpha = smoothstep(1.0, 0.95, radius);
    vec4 fillColor = stroke > 0.0 ? vec4(vec3(vColor)*0.5,alpha) : vec4(vColor.x, vColor.y, vColor.z, vColor.w * alpha);
    gl_FragColor = fog > 1.0 ? fillColor : vColor;
}
