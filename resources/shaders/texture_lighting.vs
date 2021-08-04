#version 150 core

// If we don't enable explicit_attrib/uniform_location
// then we can't do things like "layout (location = 0)"
#extension GL_ARB_explicit_attrib_location : enable
#extension GL_ARB_explicit_uniform_location : enable

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoords;
layout (location = 2) in vec3 normal;
layout (location = 11) in float shininess;
layout (location = 12) in mat4 model;

uniform mat4 projection;
//uniform mat4 model;
uniform mat4 view;
//uniform mat4 modelView;

smooth out vec3 fsPosition;
out vec3 fsNormal;
out float fsShininess;
smooth out vec2 fsTexCoords;

void main() {
    vec4 pos = model * vec4(position, 1.0);
    fsPosition = pos.xyz;
    fsTexCoords = texCoords;
    fsNormal = normalize(mat3(model) * normal);
    fsShininess = shininess;
    gl_Position = projection * view * pos;
}