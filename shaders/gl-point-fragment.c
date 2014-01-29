precision mediump float;
varying vec4 vColor;
void main(void) {
    float dx = (gl_PointCoord.x - 0.5);
    float dy = (gl_PointCoord.y - 0.5);
    float r = sqrt(dx*dx + dy*dy);
    float r1 = 0.1;

    if(r < 0.5)
        //gl_FragColor = vColor;
        gl_FragColor = 2.0 * (0.5 - r) * vColor;

    if(r >= 0.5)
        gl_FragColor[3] = 0.0;
    else if (r < r1)
        gl_FragColor[3] = vColor[3];
    else
        gl_FragColor[3] = 1.0 - ((r - r1)/(0.5 - r1))*vColor[3];
}