#version 410 core

// Cascaded Shadow Maps
// See https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping

layout (location = 0)  in vec3 position;
layout (location = 12) in mat4 model;

// out float fsTanTheta;

void main () {
	// Since dot(l, n) = cos(theta) when both are normalized, below should compute tan theta
	// fsTanTheta = tan(acos(dot(lightDir, normal)));
	gl_Position = model * vec4(position, 1.0);
}