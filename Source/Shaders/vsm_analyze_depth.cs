// Analyzes the depth buffer and determines which pages are needed

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

layout (std430, binding = VSM_PAGE_BOUNDING_BOX_BINDING_POINT) coherent buffer block5 {
    ClipMapBoundingBox clipMapBoundingBoxes[];
};

shared vec2 depthTextureSize;
shared ivec2 residencyTableSize;

const int pixelOffsets[] = int[](
    0
);

void updateResidencyStatus(in ivec2 coords, in int cascade) {
    //ivec2 pixelCoords = coords;//wrapIndex(coords, residencyTableSize);
    ivec2 pixelCoords = ivec2(
        mod(coords.x, residencyTableSize.x),
        mod(coords.y, residencyTableSize.y)
    );

    uint tileIndex = uint(pixelCoords.x + pixelCoords.y * int(numPagesXY) + cascade * int(numPagesXY * numPagesXY));
    uint pageId = 1;//computePageId(coords);

    uint frameMarker;
    uint physicalPageX;
    uint physicalPageY;
    uint memPool;
    uint prevResidencyStatus;
    uint prevDirtyBit;
    unpackPageMarkerData(
        currFramePageResidencyTable[tileIndex].info, 
        frameMarker, 
        physicalPageX,
        physicalPageY,
        memPool,
        prevResidencyStatus,
        prevDirtyBit
    );

    uint newResidencyStatus = prevResidencyStatus;
    uint newDirtyBit = prevResidencyStatus < 2 ? 1 : prevDirtyBit; 
    if (frameMarker != 1) {
        currFramePageResidencyTable[tileIndex].info = packPageMarkerData(
            1, 
            physicalPageX,
            physicalPageY,
            memPool,
            newResidencyStatus,
            newDirtyBit
        );
    }
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
    vec2 depthTexCoords = (vec2(xindex, yindex)) / depthTextureSize;
    //vec2 depthTexCoords = (vec2(xindex, yindex) + vec2(0.5)) / depthTextureSize;

    // Get current depth and convert to world space
    //float depth = textureLod(depthTexture, depthTexCoords, 0).r;
    float depth = texelFetch(depthTexture, ivec2(xindex, yindex), 0).r;

    if (depth >= 1.0) return;

    vec3 worldPosition = worldPositionFromDepth(depthTexCoords, depth, invProjectionView);

    int cascadeIndex = vsmCalculateCascadeIndexFromWorldPos(worldPosition);

    if (cascadeIndex >= vsmNumCascades) return;

    // Convert world position to a coordinate from the light's perspective

    vec3 clipCoords = vsmCalculateOriginClipValueFromWorldPos(worldPosition, cascadeIndex);
    vec3 clipCoordsLowest = vsmCalculateOriginClipValueFromWorldPos(worldPosition, int(vsmNumCascades) - 1);
    vec2 vsmTexCoords = clipCoords.xy * 0.5 + vec2(0.5);
    vec2 vsmTexCoordsLowest = clipCoordsLowest.xy * 0.5 + vec2(0.5);

    vec2 basePixelCoords = wrapIndex(vsmTexCoords, vec2(residencyTableSize));// - vec2(0.5);
    vec2 basePixelCoordsLowest = vsmTexCoordsLowest * vec2(residencyTableSize);
    // vec2 basePixelCoordsWrapped = wrapIndex(basePixelCoords, residencyTableSize);

    // float fx = fract(basePixelCoordsWrapped.x);
    // float fy = fract(basePixelCoordsWrapped.y);

    //basePixelCoords = round(basePixelCoords);

    ivec2 pixelCoordsLower = ivec2(floor(basePixelCoords));
    ivec2 pixelCoordsLowest = ivec2(floor(basePixelCoordsLowest));
    // ivec2 pixelCoordsUpper = ivec2(ceil(basePixelCoords));

    // This is a major hack to get around a strange issue where for 1 frame it seems the page table
    // memory isn't available even with GPU-driven memory management. Extending the analyze radius
    // a bit mostly gets around it, but this shouldn't be required.
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            updateResidencyStatus(pixelCoordsLower+ivec2(x,y), cascadeIndex);
        }
    }
    // Coarsest clip map
    updateResidencyStatus(pixelCoordsLowest, int(vsmNumCascades) - 1);
}