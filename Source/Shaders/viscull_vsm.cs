STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

precision highp float;
precision highp int;
precision highp uimage2D;

layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

#include "common.glsl"
#include "aabb.glsl"

uniform mat4 cascadeProjectionView;

uniform uint frameCount;
uniform uint numDrawCalls;
uniform uint maxDrawCommands;

uniform uint numPageGroupsX;
uniform uint numPageGroupsY;

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
shared int minLocalPageX;
shared int minLocalPageY;
shared int maxLocalPageX;
shared int maxLocalPageY;
shared ivec2 pageGroupCorners[4];
shared vec2 pageGroupCornersTexCoords[4];

// #define IS_PAGE_WITHIN_BOUNDS(type)                                                                 \
//     bool isPageGroupWithinBounds(in type pageCorner, in type texCornerMin, in type texCornerMax) {  \
//         return pageCorner.x >= texCornerMin.x && pageCorner.x <= texCornerMax.x &&                  \
//             pageCorner.y >= texCornerMin.y && pageCorner.y <= texCornerMax.y;                       \
//     }

#define IS_OVERLAPPING(type)                                                    \
    bool isOverlapping(in type rectAMin, in type rectAMax,                      \
                       in type rectBMin, in type rectBMax) {                    \
        return min(rectAMax.x, rectBMax.x) > max(rectAMin.x, rectBMin.x) &&     \
               min(rectAMax.y, rectBMax.y) > max(rectAMin.y, rectBMin.y);       \
    }

IS_OVERLAPPING(vec2)
IS_OVERLAPPING(ivec2)

void main() {
    ivec2 basePageGroup = ivec2(gl_WorkGroupID.xy);
    uint basePageGroupIndex = basePageGroup.x + basePageGroup.y * numPageGroupsX;

    // if (gl_LocalInvocationID == 0) {
    //     residencyTableSize = imageSize(currFramePageResidencyTable).xy;
    //     maxResidencyTableIndex = residencyTableSize - ivec2(1.0);

    //     pageCorners[0] = vec2(baseTileCoords) / vec2(maxResidencyTableIndex);
    //     pageCorners[1] = vec2(baseTileCoords + ivec2(0, 1)) / vec2(maxResidencyTableIndex);
    //     pageCorners[2] = vec2(baseTileCoords + ivec2(1, 0)) / vec2(maxResidencyTableIndex);
    //     pageCorners[3] = vec2(baseTileCoords + ivec2(1, 1)) / vec2(maxResidencyTableIndex);

    //     uint pageStatus = uint(imageLoad(currFramePageResidencyTable, baseTileCoords).r);
    //     validPage = (pageStatus == frameCount);
    // }

    // Compute residency table dimensions
    if (gl_LocalInvocationID == 0) {
        residencyTableSize = imageSize(currFramePageResidencyTable).xy;
        maxResidencyTableIndex = residencyTableSize - ivec2(1);

        minLocalPageX = residencyTableSize.x + 1;
        minLocalPageY = residencyTableSize.y + 1;
        maxLocalPageX = -1;
        maxLocalPageY = -1;
    }

    barrier();

    // Cooperatively run through the pages within this page group to check the min/max bounds
    ivec2 pagesPerPageGroup = residencyTableSize / ivec2(numPageGroupsX, numPageGroupsY);
    ivec2 startPage = basePageGroup * pagesPerPageGroup;
    ivec2 endPage = (basePageGroup + ivec2(1, 1)) * pagesPerPageGroup;
    ivec2 maxPageGroupIndex = ivec2(numPageGroupsX, numPageGroupsY) - ivec2(1);

    for (int x = startPage.x + int(gl_LocalInvocationID.x); x < endPage.x; x += int(gl_WorkGroupSize.x)) {
        for (int y = startPage.y + int(gl_LocalInvocationID.y); y < endPage.y; y += int(gl_WorkGroupSize.y)) {

            uint pageStatus = uint(imageLoad(currFramePageResidencyTable, ivec2(x, y)).r);
            if (pageStatus == frameCount) {
                atomicMin(minLocalPageX, x);
                atomicMin(minLocalPageY, y);
                atomicMax(maxLocalPageX, x + 1);
                atomicMax(maxLocalPageY, y + 1);
            }
        }
    }

    barrier();

    if (gl_LocalInvocationID == 0) {
        pageGroupCorners[0] = ivec2(minLocalPageX, minLocalPageY);
        pageGroupCorners[1] = ivec2(minLocalPageX, maxLocalPageY);
        pageGroupCorners[2] = ivec2(maxLocalPageX, minLocalPageY);
        pageGroupCorners[3] = ivec2(maxLocalPageX, maxLocalPageY);

        pageGroupCornersTexCoords[0] = vec2(pageGroupCorners[0]) / vec2(maxResidencyTableIndex);
        pageGroupCornersTexCoords[1] = vec2(pageGroupCorners[1]) / vec2(maxResidencyTableIndex);
        pageGroupCornersTexCoords[2] = vec2(pageGroupCorners[2]) / vec2(maxResidencyTableIndex);
        pageGroupCornersTexCoords[3] = vec2(pageGroupCorners[3]) / vec2(maxResidencyTableIndex);
    }

    barrier();

    // Invalid page group (all pages uncommitted)
    if (minLocalPageX > maxLocalPageX) {
        return;
    }

    // Defines local work group from layout local size tag above
    uint localWorkGroupSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y;

    vec4 corners[8];
    vec2 texCoords;
    vec2 currentTileCoords;
    vec2 currentTileCoordsLower;
    vec2 currentTileCoordsUpper;

    vec2 texCorners[4];

    for (uint drawIndex = gl_LocalInvocationIndex; drawIndex < numDrawCalls; drawIndex += localWorkGroupSize) {
        DrawElementsIndirectCommand draw = inDrawCalls[drawIndex];
        if (draw.instanceCount == 0) {
            continue;
        }

        AABB aabb = transformAabbAsNDCCoords(aabbs[drawIndex], cascadeProjectionView * modelTransforms[drawIndex]);
        computeCornersAsTexCoords(aabb, corners);

        vec2 pageMin = vec2(pageGroupCorners[0]);
        vec2 pageMax = vec2(pageGroupCorners[3]);

        vec2 aabbMin = corners[0].xy * vec2(maxResidencyTableIndex);
        vec2 aabbMax = corners[7].xy * vec2(maxResidencyTableIndex);

        if (isOverlapping(pageMin, pageMax, aabbMin, aabbMax)) {
            outDrawCalls[basePageGroupIndex * maxDrawCommands + drawIndex].instanceCount = 1;
        }

        // vec2 vmin = corners[0].xy;
        // vec2 vmax = corners[0].xy;

        // for (int i = 1; i < 8; ++i) {
        //     vmin = vec2(min(vmin.x, corners[i].x), min(vmin.y, corners[i].y));
        //     vmax = vec2(max(vmax.x, corners[i].x), max(vmax.y, corners[i].y));
        // }

        // vmin = clamp(vmin, 0.0, 1.0);
        // vmax = clamp(vmax, 0.0, 1.0);

        // Represents the texel coordinate corners of the bounding box
        // texCorners[0] = vec2(vmin.x, vmin.y);
        // texCorners[1] = vec2(vmin.x, vmax.y);
        // texCorners[2] = vec2(vmax.x, vmin.y);
        // texCorners[3] = vec2(vmax.x, vmax.y);

        // ivec2 vminPage = ivec2(vmin * vec2(maxResidencyTableIndex));
        // ivec2 vmaxPage = ivec2(vmax * vec2(maxResidencyTableIndex));

        // // Check if any part of the page is within the bounding box
        // if (isPageGroupWithinBounds(pageGroupCorners[0], vminPage, vmaxPage) ||
        //     isPageGroupWithinBounds(pageGroupCorners[1], vminPage, vmaxPage) ||
        //     isPageGroupWithinBounds(pageGroupCorners[2], vminPage, vmaxPage) ||
        //     isPageGroupWithinBounds(pageGroupCorners[3], vminPage, vmaxPage)) {

        //     //atomicExchange(outDrawCalls[basePageGroupIndex * maxDrawCommands + drawIndex].instanceCount, 1);
        //     outDrawCalls[basePageGroupIndex * maxDrawCommands + drawIndex].instanceCount = 1;
        //     continue;
        // }

        // vec2 cornerMin = vec2(minLocalPageX, minLocalPageY);
        // vec2 cornerMax = vec2(maxLocalPageX, maxLocalPageY);

        // Check if any part of the bounding box is within the page group
        // for (int j = 0; j < 4; ++j) {
        //     texCoords = texCorners[j].xy;
        //     currentTileCoords = texCoords * vec2(maxPageGroupIndex);

        //     currentTileCoordsLower = vec2(
        //         floor(currentTileCoords.x), floor(currentTileCoords.y)
        //     );

        //     currentTileCoordsUpper = vec2(
        //         ceil(currentTileCoords.x), ceil(currentTileCoords.y)
        //     );

        //     // if (isPageGroupWithinBounds(currentTileCoordsLower, cornerMin, cornerMax) ||
        //     //     isPageGroupWithinBounds(currentTileCoordsUpper, cornerMin, cornerMax)) {

        //     //     uint prev = atomicExchange(outDrawCalls[basePageGroupIndex * maxDrawCommands + drawIndex].instanceCount, 1);
        //     //     break;
        //     // }

        //     // Check if the corner lies within this tile
        //     if (currentTileCoordsLower == basePageGroup || currentTileCoordsUpper == basePageGroup) {
        //         //uint prev = atomicExchange(outDrawCalls[basePageGroupIndex * maxDrawCommands + drawIndex].instanceCount, 1);
        //         outDrawCalls[basePageGroupIndex * maxDrawCommands + drawIndex].instanceCount = 1;

        //         // if (prev == 0) {
        //         //     atomicAdd(outputDrawCalls, 1);
        //         // }

        //         break;
        //     }
        // }
    }
}