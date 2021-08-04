#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoords;

uniform mat4 projection;
uniform mat4 modelView;

smooth out vec2 fsTexCoords;

void main() {
    gl_Position = projection * modelView * vec4(position, 1.0);
    fsTexCoords = texCoords;
}