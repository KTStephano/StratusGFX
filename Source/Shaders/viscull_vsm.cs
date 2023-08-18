STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

precision highp float;
precision highp int;
precision highp uimage2D;

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#include "common.glsl"
#include "aabb.glsl"

uniform mat4 cascadeProjectionView;

uniform uint frameCount;

uniform uint numDrawCalls;

layout (r32ui) readonly uniform uimage2D currFramePageResidencyTable;

layout (std430, binding = 2) readonly buffer inputBlock2 {
    mat4 modelTransforms[];
};

layout (std430, binding = 3) readonly buffer inputBlock4 {
    AABB aabbs[];
};

layout (std430, binding = 1) readonly buffer inputBlock1 {
    DrawElementsIndirectCommand inDrawCalls[];
};

layout (std430, binding = 4) buffer outputBlock1 {
    DrawElementsIndirectCommand outDrawCalls[];
};

layout (std430, binding = 5) buffer outputBlock2 {
    int outputDrawCalls;
};

shared ivec2 residencyTableSize;
shared ivec2 maxResidencyTableIndex;
shared vec2 pageCorners[4];
shared bool validPage;

bool isPageWithinBounds(in vec2 pageCorner, in vec2 texCornerMin, in vec2 texCornerMax) {
    return pageCorner.x >= texCornerMin.x && pageCorner.x <= texCornerMax.x &&
           pageCorner.y >= texCornerMin.y && pageCorner.y <= texCornerMax.y;
}

void main() {
    ivec2 baseTileCoords = ivec2(gl_WorkGroupID.xy);

    if (gl_LocalInvocationID == 0) {
        residencyTableSize = imageSize(currFramePageResidencyTable).xy;
        maxResidencyTableIndex = residencyTableSize - ivec2(1.0);

        pageCorners[0] = vec2(baseTileCoords) / vec2(maxResidencyTableIndex);
        pageCorners[1] = vec2(baseTileCoords + ivec2(0, 1)) / vec2(maxResidencyTableIndex);
        pageCorners[2] = vec2(baseTileCoords + ivec2(1, 0)) / vec2(maxResidencyTableIndex);
        pageCorners[3] = vec2(baseTileCoords + ivec2(1, 1)) / vec2(maxResidencyTableIndex);

        uint pageStatus = uint(imageLoad(currFramePageResidencyTable, baseTileCoords).r);
        validPage = (pageStatus == frameCount);
    }

    barrier();

    // Page is not needed for this frame - skip entire work group
    if (validPage == false) {
        return;
    }

    // Defines local work group from layout local size tag above
    uint localWorkGroupSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y;

    vec4 corners[8];
    vec2 texCoords;
    vec2 currentTileCoords;
    ivec2 currentTileCoordsLower;
    ivec2 currentTileCoordsUpper;

    vec2 texCorners[4];

    for (uint i = gl_LocalInvocationIndex; i < numDrawCalls; i += localWorkGroupSize) {
        DrawElementsIndirectCommand draw = inDrawCalls[i];
        if (draw.instanceCount == 0) {
            continue;
        }

        AABB aabb = transformAabb(aabbs[i], cascadeProjectionView * modelTransforms[i]);
        computeCornersAsTexCoords(aabb, corners);

        vec2 vmin = corners[0].xy;
        vec2 vmax = corners[0].xy;

        for (int i = 1; i < 8; ++i) {
            vmin = min(vmin, corners[i].xy);
            vmax = max(vmax, corners[i].xy);
        }

        // Check if any part of the page is within the bounding box
        if (isPageWithinBounds(pageCorners[0], vmin, vmax) ||
            isPageWithinBounds(pageCorners[1], vmin, vmax) ||
            isPageWithinBounds(pageCorners[2], vmin, vmax) ||
            isPageWithinBounds(pageCorners[3], vmin, vmax)) {

            atomicExchange(outDrawCalls[i].instanceCount, 1);
            continue;
        }

        texCorners[0] = vec2(vmin.x, vmin.y);
        texCorners[1] = vec2(vmin.x, vmax.y);
        texCorners[2] = vec2(vmax.x, vmin.y);
        texCorners[3] = vec2(vmax.x, vmax.y);

        // Check if any part of the bounding box is within the page
        for (int j = 0; j < 4; ++j) {
            texCoords = texCorners[j].xy;
            currentTileCoords = texCoords * vec2(maxResidencyTableIndex);

            currentTileCoordsLower = ivec2(
                floor(currentTileCoords.x), floor(currentTileCoords.y)
            );

            currentTileCoordsUpper = ivec2(
                ceil(currentTileCoords.x), ceil(currentTileCoords.y)
            );

            // Check if the corner lies within this tile
            if (currentTileCoordsLower == baseTileCoords || currentTileCoordsUpper == baseTileCoords) {
                uint prev = atomicExchange(outDrawCalls[i].instanceCount, 1);

                // if (prev == 0) {
                //     atomicAdd(outputDrawCalls, 1);
                // }

                break;
            }
        }
    }
}