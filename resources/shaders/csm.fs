#version 410 core

// Cascaded Shadow Maps

out vec3 color;

void main() {
	// Written automatically
	// gl_FragDepth = gl_FragCoord.z
	float depth = gl_FragCoord.z;
	color = vec3(depth, depth, depth);
}