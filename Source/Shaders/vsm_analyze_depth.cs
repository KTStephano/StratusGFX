STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

precision highp float;
precision highp int;
precision highp uimage2D;
precision highp sampler2D;
precision highp sampler2DArrayShadow;

#include "common.glsl"
#include "vsm_common.glsl"

layout (local_size_x = 16, local_size_y = 9, local_size_z = 1) in;

uniform mat4 invProjectionView;

uniform sampler2D depthTexture;

uniform uint frameCount;

// layout (std430, binding = VSM_NUM_PAGES_TO_UPDATE_BINDING_POINT) buffer block1 {
//     int numPagesToMakeResident;
// };

// layout (std430, binding = VSM_PAGE_INDICES_BINDING_POINT) buffer block2 {
//     int pageIndices[];
// };

layout (std430, binding = VSM_PREV_FRAME_RESIDENCY_TABLE_BINDING) readonly buffer block3 {
    PageResidencyEntry prevFramePageResidencyTable[];
};

layout (std430, binding = VSM_CURR_FRAME_RESIDENCY_TABLE_BINDING) buffer block4 {
    PageResidencyEntry currFramePageResidencyTable[];
};

shared vec2 depthTextureSize;
shared ivec2 residencyTableSize;

void updateResidencyStatus(in ivec2 coords) {
    ivec2 pixelCoords = wrapIndex(coords, residencyTableSize);

    uint tileIndex = uint(pixelCoords.x + pixelCoords.y * int(numPagesXY));
    uint pageId = computePageId(coords);

    currFramePageResidencyTable[tileIndex].frameMarker = frameCount;

    uint prevPageId;
    uint prevDirtyBit;
    unpackPageIdAndDirtyBit(prevFramePageResidencyTable[tileIndex].info, prevPageId, prevDirtyBit);

    uint dirtyBit = prevPageId != pageId ? 1 : prevDirtyBit; 
    currFramePageResidencyTable[tileIndex].info = packPageIdWithDirtyBit(pageId, dirtyBit);
}

void main() {
    if (gl_LocalInvocationID == 0) {
        depthTextureSize = textureSize(depthTexture, 0).xy;
        //residencyTableSize = imageSize(currFramePageResidencyTable).xy;
        residencyTableSize = ivec2(numPagesXY);
    }

    barrier();

    int xindex = int(gl_GlobalInvocationID.x);
    int yindex = int(gl_GlobalInvocationID.y);

    // Depth tex coords
    vec2 depthTexCoords = (vec2(xindex, yindex) + vec2(0.5)) / depthTextureSize;
    //vec2 depthTexCoords = (vec2(xindex, yindex)) / (depthTextureSize - vec2(1.0));

    // Get current depth and convert to world space
    //float depth = textureLod(depthTexture, depthTexCoords, 0).r;
    float depth = texelFetch(depthTexture, ivec2(xindex, yindex), 0).r;

    if (depth >= 1.0) return;

    vec3 worldPosition = worldPositionFromDepth(depthTexCoords, depth, invProjectionView);

    // Convert world position to a coordinate from the light's perspective
    vec3 coords = vsmCalculateOriginClipValueFromWorldPos(worldPosition, 0);
    // Convert from range [-1, 1] to [0, 1]
    vec2 cascadeTexCoords = coords.xy * 0.5 + vec2(0.5);

    vec2 basePixelCoords = cascadeTexCoords * vec2(residencyTableSize - ivec2(1));

    float fx = fract(basePixelCoords.x);
    float fy = fract(basePixelCoords.y);

    //basePixelCoords = round(basePixelCoords);

    ivec2 pixelCoordsLower = ivec2(floor(basePixelCoords));
    ivec2 pixelCoordsUpper = ivec2(ceil(basePixelCoords));

    updateResidencyStatus(pixelCoordsLower);

    if (pixelCoordsLower != pixelCoordsUpper) {
        updateResidencyStatus(pixelCoordsUpper);
    }
}