#version 150 core

in vec3 position;

uniform mat4 projection;
uniform mat4 modelView;

out vec2 fsTexCoords;

void main() {
    gl_Position = projection * modelView * vec4(position, 1.0);
}