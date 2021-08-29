#version 410 core

// Cascaded Shadow Maps
// See https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping

layout (location = 0) in vec3 position;
layout (location = 12) in mat4 model;

uniform mat4 projection;
uniform mat4 view;

void main () {
	gl_Position = projection * view * model * vec4(position, 1.0);
}