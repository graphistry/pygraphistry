precision mediump float;

// Vertex Position in normalized range ([-1,+1])
attribute vec2 vertexPosition;

uniform float flipTexture;

varying vec2 textureCoords;

const vec2 scale = vec2(0.5, 0.5);

void main(void) {
    // scale vertex attribute to [0,1] range
    textureCoords = vertexPosition * scale + scale;

    // Flip Y coordinate if uniform says to
    if (flipTexture > 0.0) {
        textureCoords.y = 1.0 - textureCoords.y;
    }

    gl_Position = vec4(vertexPosition, 1.0, 1.0);
}
