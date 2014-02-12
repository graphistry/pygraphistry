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
uniform mat3 mvp;
attribute vec2 curPos;
// varying vec4 vColor;

void main(void) {
    float w = 1.0;

    // float maxSize = 8.0;
    // float size = maxSize * (1.0 - curPos.z);
    // if(size < 1.0) size = 1.0;
    // if(size > maxSize) size = maxSize;

    // gl_PointSize  = size;
    gl_PointSize = 3.0;

    // vec4 color = vec4(1.0, 1.0, 1.0, 1.0);
    // if(pos.x < 0.0) {
    //     color[0] = 0.0;
    // }
    // if(pos.y < 0.0) {
    //     color[1] = 0.0;
    // }
    // vColor = color;
    // vColor = vec4(1.0, 0.0, 0.0, 1.0);

    vec3 pos = mvp * vec3(curPos[0], curPos[1], w);
    gl_Position = vec4(pos[0], pos[1], 0, pos[2]);
}