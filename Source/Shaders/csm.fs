STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

precision highp float;
precision highp int;
precision highp uimage2D;
precision highp sampler2D;
precision highp sampler2DArrayShadow;

#include "common.glsl"
#include "alpha_test.glsl"
#include "atomic.glsl"
#include "vsm_common.glsl"

layout (r32ui) coherent uniform uimage2DArray vsm;

uniform uint numPagesXY;
uniform uint virtualShadowMapSizeXY;
uniform uint frameCount;
uniform ivec2 startXY;
uniform ivec2 endXY;

// Cascaded Shadow Maps
//in float fsTanTheta;
flat in int fsDrawID;
flat in int fsClipMapIndex;
smooth in vec2 fsTexCoords;
smooth in vec2 vsmTexCoords;
//smooth in float vsmDepth;

uniform float nearClipPlane;

void writeDepth(in vec2 uv, in float depth) {
	//vec2 physicalPixelCoords = wrapIndex(virtualPixelCoords, vec2(virtualShadowMapSizeXY));
	vec3 physicalPixelCoords = vec3(vsmConvertVirtualUVToPhysicalPixelCoords(
		uv,
		vec2(virtualShadowMapSizeXY),
		numPagesXY,
		fsClipMapIndex
	));

	bool resident = false;
	ivec3 physicalPixelCoordsLower = ivec3(floor(physicalPixelCoords.xy), physicalPixelCoords.z);//fsClipMapIndex);
	ivec3 physicalPixelCoordsUpper = ivec3(round(physicalPixelCoords.xy), physicalPixelCoords.z);//fsClipMapIndex);

	if (physicalPixelCoords.z >= 0) {
		IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, physicalPixelCoordsLower, depth, resident);

		// if (physicalPixelCoordsLower != physicalPixelCoordsUpper) {
		// 	IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, physicalPixelCoordsUpper, depth, resident);
		// }
	}
}

void main() {
	float vsmDepth = gl_FragCoord.z;//vsmConvertRelativeDepthToOriginDepth(gl_FragCoord.z);
	float depth = vsmDepth;//gl_FragCoord.z;

#ifdef RUN_CSM_ALPHA_TEST
	Material material = materials[materialIndices[fsDrawID]];

	vec4 baseColor = bitwiseAndBool(material.flags, GPU_DIFFUSE_MAPPED) ? texture(material.diffuseMap, fsTexCoords) : decodeMaterialData(material.diffuseColor);
	runAlphaTest(baseColor.a);

	// Written automatically - if used here it may disable early Z test but need to verify this
	//gl_FragDepth = gl_FragCoord.z;// + fsTanTheta;

	// Small offset to help prevent z fighting in certain cases
    //gl_FragDepth = baseColor.a < 1.0 ? gl_FragCoord.z - ALPHA_DEPTH_OFFSET : gl_FragCoord.z;
	//depth = baseColor.a < 1.0 ? gl_FragCoord.z - ALPHA_DEPTH_OFFSET : gl_FragCoord.z;
	depth = baseColor.a < 1.0 ? vsmDepth - ALPHA_DEPTH_OFFSET : vsmDepth;
	//depth = baseColor.a < 1.0 ? gl_FragCoord.z - ALPHA_DEPTH_OFFSET : gl_FragCoord.z;
#endif

	depth = clamp(depth, 0.0, 1.0);

	writeDepth(vsmTexCoords, depth);
}