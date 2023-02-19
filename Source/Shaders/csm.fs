STRATUS_GLSL_VERSION

// Cascaded Shadow Maps
in float fsTanTheta;
uniform float nearClipPlane;

void main() {
	// Written automatically - if used here it may disable early Z test but need to verify this
	//gl_FragDepth = gl_FragCoord.z;// + fsTanTheta;
}