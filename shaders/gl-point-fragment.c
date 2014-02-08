/*
* Copyright (C) 2011 Samsung Electronics Corporation. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided the following conditions
* are met:
*
* 1.  Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*
* 2.  Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in the
*     documentation and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY SAMSUNG ELECTRONICS CORPORATION AND ITS
* CONTRIBUTORS "AS IS", AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING
* BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SAMSUNG
* ELECTRONICS CORPORATION OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
* INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
* DATA, OR PROFITS, OR BUSINESS INTERRUPTION), HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING
* NEGLIGENCE OR OTHERWISE ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
* EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

precision mediump float;
varying vec4 vColor;
void main(void) {
    // A GL point is just a square, but we want to draw circles. Calculate if the coordinate of this
    // point (gl_PointCoord is in relation to the center of the point) is within the circle 0.5
    // units in radius around the center.
    // float dx = (gl_PointCoord.x - 0.5);
    // float dy = (gl_PointCoord.y - 0.5);
    // float r = sqrt(dx*dx + dy*dy);
    // // In between r1 and r, we fade the color out as it gets further from the center
    // float r1 = 0.1;

    // // If we're within the circle, draw a color
    // if(r < 0.5)
    //     // gl_FragColor = vColor;
    //     // Modify the pixel as we get further from the center
    //     gl_FragColor = 2.0 * (0.5 - r) * vColor;

    // // If we're outside the circle, don't draw anything
    // if(r >= 0.5)
    //     gl_FragColor[3] = 0.0;
    // // If we're within r1 of the circle, draw the pixel at full opacity
    // else if (r < r1)
    //     gl_FragColor[3] = vColor[3];
    // // If we're between r1 and r, set the alpha of this pixel to a proportionate value
    // else
    //     gl_FragColor[3] = 1.0 - ((r - r1)/(0.5 - r1))*vColor[3];

    // gl_FragColor = 2.0 * (0.5 - r) * vColor;
    gl_FragColor = vec4(1.0, 0.0, 0.0, 0.5);;
}