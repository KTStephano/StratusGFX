STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

#include "common.glsl"
#include "alpha_test.glsl"
#include "atomic.glsl"
#include "vsm_common.glsl"

layout (r32ui) coherent uniform uimage2DArray vsm;

uniform uint numPagesXY;
uniform uint virtualShadowMapSizeXY;

// Cascaded Shadow Maps
//in float fsTanTheta;
flat in int fsDrawID;
smooth in vec2 fsTexCoords;
smooth in vec2 vsmTexCoords;
smooth in float vsmDepth;

uniform float nearClipPlane;

void main() {
	float depth = vsmDepth;//gl_FragCoord.z;

#ifdef RUN_CSM_ALPHA_TEST
	Material material = materials[materialIndices[fsDrawID]];

	vec4 baseColor = bitwiseAndBool(material.flags, GPU_DIFFUSE_MAPPED) ? texture(material.diffuseMap, fsTexCoords) : FLOAT4_TO_VEC4(material.diffuseColor);
	runAlphaTest(baseColor.a);

	// Written automatically - if used here it may disable early Z test but need to verify this
	//gl_FragDepth = gl_FragCoord.z;// + fsTanTheta;

	// Small offset to help prevent z fighting in certain cases
    //gl_FragDepth = baseColor.a < 1.0 ? gl_FragCoord.z - ALPHA_DEPTH_OFFSET : gl_FragCoord.z;
	//depth = baseColor.a < 1.0 ? gl_FragCoord.z - ALPHA_DEPTH_OFFSET : gl_FragCoord.z;
	depth = baseColor.a < 1.0 ? vsmDepth - ALPHA_DEPTH_OFFSET : vsmDepth;
	//depth = baseColor.a < 1.0 ? gl_FragCoord.z - ALPHA_DEPTH_OFFSET : gl_FragCoord.z;
#endif

	//ivec3 vsmCoords = ivec3(gl_FragCoord.xy, 0);
	vec3 vsmCoords = vec3(vsmTexCoords * (vec2(virtualShadowMapSizeXY) - vec2(1.0)), 0.0);
	vsmCoords.xy = wrapIndex(vsmCoords.xy, vec2(virtualShadowMapSizeXY));

	IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, ivec3(vsmCoords), depth);
	IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, ivec3(vsmCoords) + ivec3(0, 1, 0), depth);
	IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, ivec3(vsmCoords) + ivec3(0, -1, 0), depth);
	IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, ivec3(vsmCoords) + ivec3(1, 0, 0), depth);
	IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, ivec3(vsmCoords) + ivec3(-1, 0, 0), depth);
	//imageStore(vsm, vsmCoords, uvec4(floatBitsToUint(1.0)));

	//imageAtomicExchange(vsm, vsmCoordsLower, floatBitsToUint(depth));
	//imageAtomicMin(vsm, vsmCoordsLower, floatBitsToUint(depth));
	//imageAtomicMin(vsm, vsmCoordsUpper, floatBitsToUint(depth));
	//imageStore(vsm, vsmCoords, uvec4(floatBitsToUint(0.0)));
}