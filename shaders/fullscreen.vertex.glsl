precision mediump float;

// Vertex Position in normalized range ([-1,+1])
attribute vec2 vertexPosition;

varying vec2 textureCoords;

const vec2 scale = vec2(0.5, 0.5);

void main(void) {

    textureCoords = vertexPosition * scale + scale; // scale vertex attribute to [0,1] range
    gl_Position = vec4(vertexPosition, 1.0, 1.0);
}
