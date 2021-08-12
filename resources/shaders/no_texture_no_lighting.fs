#version 410 core

in vec2 fsTexCoords;

uniform vec3 diffuseColor;
out vec4 color;

void main() {
    color = vec4(diffuseColor, 1.0);
}