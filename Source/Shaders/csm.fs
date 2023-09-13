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

	// ivec2 physicalPageCoords = ivec2(physicalPixelCoords / vec2(VSM_MAX_NUM_TEXELS_PER_PAGE_XY));
	// //ivec2 physicalPageCoords = ivec2(round(wrapIndex(vsmTexCoords * vec2(numPagesXY - 1), vec2(numPagesXY))));
	// uint physicalPageIndex = physicalPageCoords.x + physicalPageCoords.y * numPagesXY + uint(fsClipMapIndex) * numPagesXY * numPagesXY;

	// PageResidencyEntry entry = currFramePageResidencyTable[physicalPageIndex];

	// uint pageId;
	// uint dirtyBit;
	// unpackPageIdAndDirtyBit(entry.info, pageId, dirtyBit);

	bool resident = false;
	ivec3 physicalPixelCoordsLower = ivec3(floor(physicalPixelCoords.xy), physicalPixelCoords.z);//fsClipMapIndex);
	ivec3 physicalPixelCoordsUpper = ivec3(ceil(physicalPixelCoords.xy), physicalPixelCoords.z);//fsClipMapIndex);

	// uint frameMarker;
	// uint unused;
	// unpackFrameCountAndUpdateCount(
	// 	entry.frameMarker,
	// 	frameMarker,
	// 	unused
	// );

	//if (dirtyBit > 0 && entry.frameMarker == frameCount) {
	//if (frameMarker == frameCount) {
	//if (dirtyBit > 0) {
	//if (physicalPageCoords.x >= startXY.x && physicalPageCoords.x <= endXY.x &&
	//	physicalPageCoords.y >= startXY.y && physicalPageCoords.y <= endXY.y) {

	if (physicalPixelCoords.z >= 0) {
		IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, physicalPixelCoordsLower, depth, resident);
	}
		//IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, physicalPixelCoordsUpper, depth, resident);
		// if (!resident) {
		// 	IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, physicalPixelCoordsUpper, depth, resident);
		// }
		
		// if (dirtyBit > 0 && dirtyBit != VSM_PAGE_RENDERED_BIT) {
		// 	uint newDirtyBit = VSM_PAGE_CLEARED_BIT;
		// 	currFramePageResidencyTable[physicalPageIndex].info = packPageIdWithDirtyBit(pageId, newDirtyBit);
		// }
		//IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, physicalPixelCoordsUpper, depth);
		//IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, physicalPixelCoordsLower + ivec3(1, 1, 0), depth);
		// if (physicalPixelCoordsLower != physicalPixelCoordsUpper) {
		// 	IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, physicalPixelCoordsUpper, depth);
		// }
	//}
}

// void markPage(in vec2 physicalPixelCoords) {
// 	ivec2 physicalPageCoords = ivec2(physicalPixelCoords / vec2(VSM_MAX_NUM_TEXELS_PER_PAGE_XY));
// 	uint physicalPageIndex = physicalPageCoords.x + physicalPageCoords.y * numPagesXY + uint(fsClipMapIndex) * numPagesXY * numPagesXY;

// 	PageResidencyEntry entry = currFramePageResidencyTable[physicalPageIndex];

// 	// uint frameMarker;
// 	// uint unused;
// 	// unpackFrameCountAndUpdateCount(
// 	// 	entry.frameMarker,
// 	// 	frameMarker,
// 	// 	unused
// 	// );

// 	uint pageId;
// 	uint dirtyBit;
// 	unpackPageIdAndDirtyBit(entry.info, pageId, dirtyBit);

// 	if (dirtyBit > 0 && dirtyBit != VSM_PAGE_RENDERED_BIT) {
// 		uint newDirtyBit = VSM_PAGE_CLEARED_BIT;
// 		currFramePageResidencyTable[physicalPageIndex].info = packPageIdWithDirtyBit(pageId, newDirtyBit);
// 	}
// }

// void writeDepth(in vec2 virtualPixelCoords, in float depth) {
// 	vec2 physicalPixelCoords = wrapIndex(virtualPixelCoords, vec2(virtualShadowMapSizeXY));

// 	//markPage(floor(physicalPixelCoords));
// 	// markPage(floor(physicalPixelCoords) + vec2(1, 0));
// 	// markPage(floor(physicalPixelCoords) + vec2(-1, 0));
// 	// markPage(floor(physicalPixelCoords) + vec2(0, 1));
// 	// markPage(floor(physicalPixelCoords) + vec2(0, -1));

// 	// int offset = 2;
// 	// for (int x = -offset; x <= offset; ++x) {
// 	// 	for (int y = -offset; y <= offset; ++y) {
// 	// 		markPage(floor(physicalPixelCoords) + vec2(x, y));
// 	// 	}
// 	// }

// 	ivec3 physicalPixelCoordsLower = ivec3(floor(physicalPixelCoords.xy), fsClipMapIndex);
// 	ivec3 physicalPixelCoordsUpper = ivec3(round(physicalPixelCoords.xy), fsClipMapIndex);

// 	IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, physicalPixelCoordsLower, depth);
// 	//IMAGE_ATOMIC_MIN_FLOAT_SPARSE(vsm, physicalPixelCoordsUpper, depth);
// }

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

	//ivec3 vsmCoords = ivec3(gl_FragCoord.xy, 0);
	//vec2 virtualPixelCoords = vec2(gl_FragCoord.xy);
	//vec2 physicalPixelCoords = convertVirtualCoordsToPhysicalCoords(virtualPixelCoords, vec2(imageSize(vsm).xy) - 1, fsClipMapIndex);
	//vec3 vsmCoords = vec3(vsmTexCoords * (vec2(virtualShadowMapSizeXY) - vec2(1.0)), 0.0);

	//vec3 vsmCoords = vec3(vsmTexCoords * vec2(imageSize(vsm).xy) - vec2(0.5), 0.0);
	//vec3 vsmCoords = vec3(vsmTexCoords * vec2(imageSize(vsm).xy), 0.0);

	//vec3 vsmCoords = vec3(physicalPixelCoords, 0.0);
	//vsmCoords.xy = wrapIndex(vsmCoords.xy, vec2(virtualShadowMapSizeXY));
	//vec3 vsmCoords = vec3(virtualPixelCoords, 0.0);

	writeDepth(vsmTexCoords, depth);

    // float fx = fract(vsmCoords.x);
    // float fy = fract(vsmCoords.y);

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