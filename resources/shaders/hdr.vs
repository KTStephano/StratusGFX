#version 410 core
// If we don't enable explicit_attrib/uniform_location
// then we can't do things like "layout (location = 0)"
#extension GL_ARB_explicit_attrib_location : enable
#extension GL_ARB_explicit_uniform_location : enable

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoords;

out vec2 fsTexCoords;

void main() {
    fsTexCoords = texCoords;
    gl_Position = vec4(position, 1.0);
}