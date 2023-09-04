STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

precision highp float;
precision highp int;
precision highp uimage2D;
precision highp sampler2D;
precision highp sampler2DArrayShadow;

layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

#include "common.glsl"
#include "aabb.glsl"
#include "vsm_common.glsl"

// uniform mat4 cascadeProjectionView;
// uniform mat4 invCascadeProjectionView;
// uniform mat4 vsmProjectionView;

uniform uint frameCount;
uniform uint numDrawCalls;
uniform uint maxDrawCommands;

uniform uint numPageGroupsX;
uniform uint numPageGroupsY;
uniform uint numPagesXY;
uniform uint numPixelsXY;

layout (std430, binding = CURR_FRAME_MODEL_MATRICES_BINDING_POINT) readonly buffer inputBlock2 {
    mat4 modelTransforms[];
};

layout (std430, binding = AABB_BINDING_POINT) readonly buffer inputBlock4 {
    AABB aabbs[];
};

layout (std430, binding = VISCULL_VSM_IN_DRAW_CALLS_BINDING_POINT) readonly buffer inputBlock1 {
    DrawElementsIndirectCommand inDrawCalls[];
};

layout (std430, binding = VISCULL_VSM_OUT_DRAW_CALLS_BINDING_POINT) buffer outputBlock1 {
    DrawElementsIndirectCommand outDrawCalls[];
};

// layout (std430, binding = 5) buffer outputBlock2 {
//     int outputDrawCalls;
// };

// layout (std430, binding = VSM_PAGE_GROUPS_TO_RENDER_BINDING_POINT) buffer outputBlock3 {
//     uint pageGroupsToRender[];
// };

layout (std430, binding = VSM_CURR_FRAME_RESIDENCY_TABLE_BINDING) readonly buffer inputBlock3 {
    PageResidencyEntry currFramePageResidencyTable[];
};

layout (std430, binding = VSM_PAGE_BOUNDING_BOX_BINDING_POINT) readonly buffer inputBlock5 {
    int minPageX;
    int minPageY;
    int maxPageX;
    int maxPageY;
};

shared ivec2 pageGroupCorners[4];
shared vec2 pageGroupCornersTexCoords[4];
// shared uint pageGroupIsValid;
// shared uint atLeastOneResidentPage;

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

    if (minPageX > maxPageX || minPageY > maxPageY) {
        return;
    }

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

    ivec2 residencyTableSize = ivec2(numPagesXY, numPagesXY);
    ivec2 texelsPerPage = ivec2(numPixelsXY) / residencyTableSize;
    ivec2 maxResidencyTableIndex = residencyTableSize - ivec2(1);
    ivec2 pagesPerPageGroup = residencyTableSize / ivec2(numPageGroupsX, numPageGroupsY);
    ivec2 maxPageGroupIndex = ivec2(numPageGroupsX, numPageGroupsY) - ivec2(1);

    // ivec2(2, 2) is so we can add a one page border to account for times when this
    // virtual page group does not align perfectly with the physical page group
    ivec2 baseStartPage = basePageGroup * pagesPerPageGroup;
    //ivec2 baseEndPage = (basePageGroup + ivec2(2, 2)) * pagesPerPageGroup;
    ivec2 baseEndPage = (basePageGroup + ivec2(1)) * pagesPerPageGroup;

    // ivec2 startPage = baseStartPage - ivec2(1, 1);
    // ivec2 endPage = baseEndPage + ivec2(1, 1);
    ivec2 startPage = baseStartPage - ivec2(1);
    ivec2 endPage = baseEndPage + ivec2(1);

    if (gl_LocalInvocationID == 0) {
        pageGroupCorners[0] = ivec2(minPageX, minPageY);
        pageGroupCorners[1] = ivec2(minPageX, maxPageY);
        pageGroupCorners[2] = ivec2(maxPageX, minPageY);
        pageGroupCorners[3] = ivec2(maxPageX, maxPageY);

        pageGroupCornersTexCoords[0] = vec2(pageGroupCorners[0]) / vec2(maxResidencyTableIndex);
        pageGroupCornersTexCoords[1] = vec2(pageGroupCorners[1]) / vec2(maxResidencyTableIndex);
        pageGroupCornersTexCoords[2] = vec2(pageGroupCorners[2]) / vec2(maxResidencyTableIndex);
        pageGroupCornersTexCoords[3] = vec2(pageGroupCorners[3]) / vec2(maxResidencyTableIndex);
    }

    barrier();

    // Invalid page group (all pages uncommitted)
    // if (atLeastOneResidentPage == 0 || atLeastOneResidentPage == 0) {
    //     return;
    // }

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
        // if (draw.instanceCount == 0) {
        //     continue;
        // }

        //AABB aabb = transformAabbAsNDCCoords(aabbs[drawIndex], cascadeProjectionView * modelTransforms[drawIndex]);
        AABB aabb = transformAabbAsNDCCoords(aabbs[drawIndex], vsmClipMap0ProjectionView * modelTransforms[drawIndex]);
        computeCornersAsTexCoords(aabb, corners);

        vec2 pageMin = vec2(pageGroupCorners[0]);
        vec2 pageMax = vec2(pageGroupCorners[3]);

        vec2 aabbMin = corners[0].xy * vec2(maxResidencyTableIndex);
        vec2 aabbMax = corners[7].xy * vec2(maxResidencyTableIndex);

        // Even if our page group is inactive we still need to record commands just in case
        // our inactivity is due to being fully cached (the CPU may clear some/all of our region
        // due to its conservative algorithm)
        if (isOverlapping(pageMin, pageMax, aabbMin, aabbMax)) {
            draw.instanceCount = 1;

            //outDrawCalls[basePageGroupIndex * maxDrawCommands + drawIndex].instanceCount = 1;
            //atomicExchange(outDrawCalls[drawIndex].instanceCount, 1);
            // outDrawCalls[drawIndex].instanceCount = 1;

            // Mark this page group as valid for this frame
            //atomicOr(pageGroupsToRender[basePageGroupIndex], frameCount);
            // atomicOr(pageGroupsToRender[basePageGroupIndex], pageGroupIsValid);
        }
        else {
            draw.instanceCount = 0;
        }

        outDrawCalls[drawIndex] = draw;
    }
}