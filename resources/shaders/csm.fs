#version 400 core

// Cascaded Shadow Maps
in float fsTanTheta;

void main() {
	// Written automatically
	gl_FragDepth = gl_FragCoord.z;// + fsTanTheta;
}