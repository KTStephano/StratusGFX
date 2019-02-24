#version 150 core

in vec3 position;
in vec2 texCoords;
in vec3 normal;

uniform mat4 projection;
uniform mat4 model;
uniform mat4 view;
//uniform mat4 modelView;

smooth out vec3 fsPosition;
out vec3 fsNormal;
smooth out vec2 fsTexCoords;

void main() {
    vec4 pos = model * vec4(position, 1.0);
    fsPosition = pos.xyz;
    fsTexCoords = texCoords;
    fsNormal = normalize(mat3(model) * normal);
    gl_Position = projection * view * pos;
}