STRATUS_GLSL_VERSION

// Cascaded Shadow Maps
in float fsTanTheta;

void main() {
	// Written automatically
	gl_FragDepth = gl_FragCoord.z;// + fsTanTheta;
}