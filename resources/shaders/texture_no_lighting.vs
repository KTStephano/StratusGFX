#version 150 core

in vec3 position;
in vec2 texCoords;

uniform mat4 projection;
uniform mat4 modelView;

smooth out vec2 fsTexCoords;

void main() {
    gl_Position = projection * modelView * vec4(position, 1.0);
    fsTexCoords = texCoords;
}