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
#ifdef RUN_CSM_ALPHA_TEST
	Material material = materials[materialIndices[fsDrawID]];
	vec4 baseColor = FLOAT4_TO_VEC4(material.diffuseColor);

    if (bitwiseAndBool(material.flags, GPU_DIFFUSE_MAPPED)) {
        baseColor = texture(material.diffuseMap, fsTexCoords);
    }

	runAlphaTest(baseColor.a);

	// Written automatically - if used here it may disable early Z test but need to verify this
	//gl_FragDepth = gl_FragCoord.z;// + fsTanTheta;

	// Small offset to help prevent z fighting in certain cases
    if (baseColor.a < 1.0) {
        gl_FragDepth = gl_FragCoord.z - ALPHA_DEPTH_OFFSET;
    }
    else {
        gl_FragDepth = gl_FragCoord.z;
    }
#endif
}