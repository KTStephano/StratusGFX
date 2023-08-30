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

layout (std430, binding = VSM_CURR_FRAME_RESIDENCY_TABLE_BINDING) readonly buffer block1 {
    PageResidencyEntry currFramePageResidencyTable[];
};

uniform uint numPagesXY;
uniform uint virtualShadowMapSizeXY;
uniform uint frameCount;

// Cascaded Shadow Maps
//in float fsTanTheta;
flat in int fsDrawID;
smooth in vec2 fsTexCoords;
smooth in vec2 vsmTexCoords;
smooth in float vsmDepth;

uniform float nearClipPlane;

void writeDepth(in vec2 virtualPixelCoords, in float depth) {
	vec2 physicalPixelCoords = wrapIndex(virtualPixelCoords, vec2(virtualShadowMapSizeXY));
	ivec2 physicalPageCoords = ivec2(physicalPixelCoords / vec2(VSM_MAX_NUM_TEXELS_PER_PAGE_XY));
	uint physicalPageIndex = physicalPageCoords.x + physicalPageCoords.y * numPagesXY;

	PageResidencyEntry entry = currFramePageResidencyTable[physicalPageIndex];

	uint pageId;
	uint dirtyBit;
	unpackPageIdAndDirtyBit(entry.info, pageId, dirtyBit);

	if (dirtyBit > 0 && entry.frameMarker == frameCount) {
		IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, ivec3(floor(physicalPixelCoords.xy), 0.0), depth);
		IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, ivec3(ceil(physicalPixelCoords.xy), 0.0), depth);
	}
}

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
	//vsmCoords.xy = wrapIndex(vsmCoords.xy, vec2(virtualShadowMapSizeXY));

	writeDepth(vsmCoords.xy, depth);

    float fx = fract(vsmCoords.x);
    float fy = fract(vsmCoords.y);

	//vsmCoords = ceil(vsmCoords);

	// If we are approaching a texel boundary then allocate a bit of the region around us
    // if (fx <= 0.25) {
	// 	writeDepth(vsmCoords.xy + vec2(-1, 0), depth);
    // }
    // else if (fx >= 0.75) {
	// 	writeDepth(vsmCoords.xy + vec2(1, 0), depth);
    // }

    // if (fy <= 0.25) {
	// 	writeDepth(vsmCoords.xy + vec2(0, -1), depth);
    // }
    // else if (fy >= 0.75) {
	// 	writeDepth(vsmCoords.xy + vec2(0, 1), depth);
    // }

	// writeDepth(vsmCoords.xy + vec2(0, 1), depth);
	// writeDepth(vsmCoords.xy + vec2(0, -1), depth);
	// writeDepth(vsmCoords.xy + vec2(1, 0), depth);
	// writeDepth(vsmCoords.xy + vec2(-1, 0), depth);

	// IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, ivec3(floor(vsmCoords)) + ivec3(0, 1, 0), depth);
	// IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, ivec3(floor(vsmCoords)) + ivec3(0, -1, 0), depth);
	// IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, ivec3(floor(vsmCoords)) + ivec3(1, 0, 0), depth);
	// IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, ivec3(floor(vsmCoords)) + ivec3(-1, 0, 0), depth);
}