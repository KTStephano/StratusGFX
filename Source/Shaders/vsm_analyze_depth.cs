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

// uniform mat4 cascadeProjectionView;
uniform mat4 invProjectionView;

uniform sampler2D depthTexture;

uniform uint frameCount;
uniform uint numPagesXY;

// layout (std430, binding = VSM_NUM_PAGES_TO_UPDATE_BINDING_POINT) buffer block1 {
//     int numPagesToMakeResident;
// };

// layout (std430, binding = VSM_PAGE_INDICES_BINDING_POINT) buffer block2 {
//     int pageIndices[];
// };

// layout (std430, binding = VSM_PREV_FRAME_RESIDENCY_TABLE_BINDING) readonly buffer block3 {
//     PageResidencyEntry prevFramePageResidencyTable[];
// };

layout (std430, binding = VSM_CURR_FRAME_RESIDENCY_TABLE_BINDING) coherent buffer block4 {
    PageResidencyEntry currFramePageResidencyTable[];
};

layout (std430, binding = VSM_PAGE_BOUNDING_BOX_BINDING_POINT) buffer block5 {
    ClipMapBoundingBox clipMapBoundingBoxes[];
};

shared vec2 depthTextureSize;
shared ivec2 residencyTableSize;

// const int pixelOffsets[] = int[](
//     -2, -1, 0, 1, 2
// );
// const int pixelOffsets[] = int[](
//     -1, 0, 1
// );
const int pixelOffsets[] = int[](
    0
);

void updateResidencyStatus(in ivec2 coords, in int cascade) {
    ivec2 pixelCoords = wrapIndex(coords, residencyTableSize);

    uint tileIndex = uint(pixelCoords.x + pixelCoords.y * int(numPagesXY) + cascade * int(numPagesXY * numPagesXY));
    uint pageId = computePageId(coords);

    uint prevPageId;
    uint prevDirtyBit;
    unpackPageIdAndDirtyBit(currFramePageResidencyTable[tileIndex].info, prevPageId, prevDirtyBit);

    uint unused;
    uint prevUpdateCount;
    unpackFrameCountAndUpdateCount(currFramePageResidencyTable[tileIndex].frameMarker, unused, prevUpdateCount);

    uint newUpdateCount = prevUpdateCount;
    currFramePageResidencyTable[tileIndex].frameMarker = packFrameCountWithUpdateCount(frameCount, newUpdateCount);

    // if (prevDirtyBit == VSM_PAGE_RENDERED_BIT) {
    //     prevDirtyBit = 0;
    // }

    uint dirtyBit = (prevPageId != pageId || prevUpdateCount < 2) ? 1 : prevDirtyBit; 
    currFramePageResidencyTable[tileIndex].info = packPageIdWithDirtyBit(pageId, dirtyBit);
}

void main() {
    if (gl_LocalInvocationID == 0) {
        depthTextureSize = textureSize(depthTexture, 0).xy;
        //residencyTableSize = imageSize(currFramePageResidencyTable).xy;
        residencyTableSize = ivec2(numPagesXY);

        for (int i = 0; i < vsmNumCascades; ++i) {
            clipMapBoundingBoxes[i].minPageX = int(numPagesXY) + 1;
            clipMapBoundingBoxes[i].minPageY = int(numPagesXY) + 1;
            clipMapBoundingBoxes[i].maxPageX = -1;
            clipMapBoundingBoxes[i].maxPageY = -1;
        }
    }

    barrier();

    int xindex = int(gl_GlobalInvocationID.x);
    int yindex = int(gl_GlobalInvocationID.y);

    // Depth tex coords
    vec2 depthTexCoords = (vec2(xindex, yindex) + vec2(0.5)) / depthTextureSize;
    //vec2 depthTexCoords = (vec2(xindex, yindex) + vec2(1.0)) / (depthTextureSize);

    // Get current depth and convert to world space
    //float depth = textureLod(depthTexture, depthTexCoords, 0).r;
    float depth = texelFetch(depthTexture, ivec2(xindex, yindex), 0).r;

    if (depth >= 1.0) return;

    vec3 worldPosition = worldPositionFromDepth(depthTexCoords, depth, invProjectionView);

    int cascadeIndex = vsmCalculateCascadeIndexFromWorldPos(worldPosition);

    if (cascadeIndex >= vsmNumCascades) return;

    // Convert world position to a coordinate from the light's perspective
    // vec4 coords = cascadeProjectionView * vec4(worldPosition, 1.0);
    // vec2 cascadeTexCoords = coords.xy / coords.w; // Perspective divide
    // // Convert from range [-1, 1] to [0, 1]
    // cascadeTexCoords.xy = cascadeTexCoords.xy * 0.5 + vec2(0.5);

    vec3 clipCoords = vsmCalculateOriginClipValueFromWorldPos(worldPosition, cascadeIndex);
    vec2 vsmTexCoords = clipCoords.xy * 0.5 + vec2(0.5);

    vec2 basePixelCoords = vsmTexCoords * vec2(residencyTableSize);// - vec2(0.5);
    vec2 basePixelCoordsWrapped = wrapIndex(basePixelCoords, residencyTableSize);

    float fx = fract(basePixelCoordsWrapped.x);
    float fy = fract(basePixelCoordsWrapped.y);

    //basePixelCoords = round(basePixelCoords);

    ivec2 pixelCoordsLower = ivec2(floor(basePixelCoords));
    ivec2 pixelCoordsUpper = ivec2(ceil(basePixelCoords));

    updateResidencyStatus(pixelCoordsLower, cascadeIndex);
    updateResidencyStatus(pixelCoordsUpper, cascadeIndex);

    // ivec2 coords1 = pixelCoordsLower;
    // ivec2 coords2 = ivec2(pixelCoordsLower.x, pixelCoordsUpper.y);
    // ivec2 coords3 = ivec2(pixelCoordsUpper.x, pixelCoordsLower.y);
    // ivec2 coords4 = pixelCoordsUpper;

    // if (fx <= 0.02 || fx >= 0.98 || fy <= 0.02 || fy >= 0.98) {
    //     int offset = 1;
    //     for (int x = -offset; x <= offset; ++x) {
    //         for (int y = -offset; y <= offset; ++y) {
    //             updateResidencyStatus(pixelCoordsLower + ivec2(x, y), cascadeIndex);
    //         }
    //     }
    // }
    // else {
    //     updateResidencyStatus(pixelCoordsLower, cascadeIndex);
    // }

    // if (coords2 != coords1) {
    //     updateResidencyStatus(coords2, cascadeIndex);
    // }
    // if (coords3 != coords2) {
    //     updateResidencyStatus(coords3, cascadeIndex);
    // }
    // if (coords4 != coords3) {
    //     updateResidencyStatus(coords4, cascadeIndex);
    // }

    // Check for approaching page boundaries
    // if (fx <= 0.01) {
    //     updateResidencyStatus(pixelCoordsLower + ivec2(-1, 0), cascadeIndex);
    //     // updateResidencyStatus(pixelCoordsLower + ivec2(-2, 0), cascadeIndex);

    //     if (fy <= 0.01) {
    //         updateResidencyStatus(pixelCoordsLower + ivec2(-1, -1), cascadeIndex);
    //     }
    //     else if (fy >= 0.98) {
    //         updateResidencyStatus(pixelCoordsLower + ivec2(-1, 1), cascadeIndex);
    //     }
    // }
    // else if (fx >= 0.98) {
    //     updateResidencyStatus(pixelCoordsLower + ivec2(1, 0), cascadeIndex);
    //     // updateResidencyStatus(pixelCoordsLower + ivec2(2, 0), cascadeIndex);
    //     if (fy <= 0.01) {
    //         updateResidencyStatus(pixelCoordsLower + ivec2(1, -1), cascadeIndex);
    //     }
    //     else if (fy >= 0.98) {
    //         updateResidencyStatus(pixelCoordsLower + ivec2(1, 1), cascadeIndex);
    //     }
    // }

    // if (fy <= 0.01) {
    //     updateResidencyStatus(pixelCoordsLower + ivec2(0, -1), cascadeIndex);
    //     // updateResidencyStatus(pixelCoordsLower + ivec2(0, -2), cascadeIndex);
    // }
    // else if (fy >= 0.98) {
    //     updateResidencyStatus(pixelCoordsLower + ivec2(0, 1), cascadeIndex);
    //     // updateResidencyStatus(pixelCoordsLower + ivec2(0, 2), cascadeIndex);
    // }

    // int offset = 1;
    // for (int x = -offset; x <= offset; ++x) {
    //     for (int y = -offset; y <= offset; ++y) {
    //         updateResidencyStatus(pixelCoordsLower + ivec2(x, y), cascadeIndex);
    //     }
    // }

    //updateResidencyStatus(pixelCoordsLower, cascadeIndex);
    // if (pixelCoordsLower != pixelCoordsUpper) {
    //     updateResidencyStatus(pixelCoordsUpper, cascadeIndex);
    // }
}