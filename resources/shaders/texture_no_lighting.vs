#version 150 core

in vec3 position;
in vec2 texCoords;

uniform in mat4 modelViewProjection;

out vec2 fsTexCoords;

void main() {
    gl_Position = modelViewProjection * vec4(position, 1.0);
    fsTexCoords = texCoords;
}