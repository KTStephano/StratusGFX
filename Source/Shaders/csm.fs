STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "common.glsl"
#include "alpha_test.glsl"

// Cascaded Shadow Maps
in float fsTanTheta;
flat in int fsDrawID;
smooth in vec2 fsTexCoords;

uniform float nearClipPlane;

void main() {
	// Material material = materials[materialIndices[fsDrawID]];
	// vec4 baseColor = material.diffuseColor;

    // if (bitwiseAndBool(material.flags, GPU_DIFFUSE_MAPPED)) {
    //     baseColor = texture(material.diffuseMap, fsTexCoords);
    // }

	// runAlphaTest(baseColor.a, 0.25);

	// Written automatically - if used here it may disable early Z test but need to verify this
	//gl_FragDepth = gl_FragCoord.z;// + fsTanTheta;
}