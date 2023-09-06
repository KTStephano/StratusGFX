STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

precision highp float;
precision highp int;
precision highp uimage2D;
precision highp sampler2D;
precision highp sampler2DArrayShadow;

#include "common.glsl"
#include "vsm_common.glsl"

layout (r32ui) coherent uniform uimage2DArray vsm;

layout (std430, binding = VSM_CURR_FRAME_RESIDENCY_TABLE_BINDING) coherent buffer block1 {
    PageResidencyEntry currFramePageResidencyTable[];
};

uniform float clearValue = 1.0;
uniform uint frameCount;
uniform int vsmClipMapIndex;
uniform ivec2 numPagesXY;
uniform ivec2 startXY;
uniform ivec2 endXY;

shared uint clearValueBits;
shared ivec2 vsmSize;
shared ivec2 vsmMaxIndex;
shared ivec2 vsmPixelStart;
shared ivec2 vsmPixelEnd;
shared uint frameMarker;
shared uint updateCount;

// void clearPixel(in vec2 physicalPixelCoords) {
//     // float fx = fract(physicalPixelCoords.x);
//     // float fy = fract(physicalPixelCoords.y);
//     //vec2 physicalPixelCoords = vec2(virtualPixelCoords);

//     imageStore(vsm, ivec3(ivec2(physicalPixelCoords), 0), uvec4(clearValueBits));
//     // imageStore(vsm, ivec3(ceil(physicalPixelCoords), 0), uvec4(clearValueBits));

//     // TODO: Figure out why dividing physicalPixelCoords by numPagesXY gave the wrong answer
//     vec2 physicalPageTexCoords = vec2(physicalPixelCoords) / vec2(vsmMaxIndex);
//     vec2 physicalPageCoords = physicalPageTexCoords * (vec2(numPagesXY) - vec2(1.0));
//     //ivec2 physicalPageCoordsRounded = ivec2(roundCoords(physicalPageCoords, vec2(numPagesXY) - vec2(1.0)));
//     ivec2 physicalPageCoordsRounded = ivec2(ceil(physicalPageCoords));
//     atomicAnd(currFramePageResidencyTable[physicalPageCoordsRounded.x + physicalPageCoordsRounded.y * numPagesXY.x].info, VSM_PAGE_ID_MASK);
//     // atomicAnd(currFramePageResidencyTable[physicalPageCoordsUpper.x + physicalPageCoordsUpper.y * numPagesXY.x].info, VSM_PAGE_ID_MASK);
// }

void clearPixel(in ivec2 physicalPixelCoords) {
    imageStore(vsm, ivec3(physicalPixelCoords, vsmClipMapIndex), uvec4(clearValueBits));

    // vec2 virtualPixelCoords = convertPhysicalCoordsToVirtualCoords(
    //     physicalPixelCoords,
    //     vsmMaxIndex,
    //     vsmClipMapIndex
    // );
}

void main() {
    if (gl_LocalInvocationID == 0) {
        clearValueBits = floatBitsToUint(clearValue);
        vsmSize = imageSize(vsm).xy;
        vsmMaxIndex = vsmSize - ivec2(1.0);
    }

    barrier();

    uvec2 physicalPageCoords = gl_WorkGroupID.xy;
    uint physicalPageIndex = uint(physicalPageCoords.x + physicalPageCoords.y * numPagesXY.x + vsmClipMapIndex * numPagesXY.x * numPagesXY.y);

    if (gl_LocalInvocationID == 0) {
        vsmPixelStart = ivec2(physicalPageCoords.x * VSM_MAX_NUM_TEXELS_PER_PAGE_XY, physicalPageCoords.y * VSM_MAX_NUM_TEXELS_PER_PAGE_XY);
        vsmPixelEnd = vsmPixelStart + ivec2(VSM_MAX_NUM_TEXELS_PER_PAGE_XY);
        unpackFrameCountAndUpdateCount(
            currFramePageResidencyTable[physicalPageIndex].frameMarker,
            frameMarker,
            updateCount
        );
    }

    uint pageId;
    uint dirtyBit;
    unpackPageIdAndDirtyBit(
        currFramePageResidencyTable[physicalPageIndex].info, 
        pageId,
        dirtyBit
    );

    barrier();

    uint updatedDirtyBit = VSM_PAGE_CLEARED_BIT;

    if (gl_LocalInvocationID == 0 && dirtyBit > 0 && frameMarker == frameCount) {
        vec2 virtualPageCoords = convertPhysicalCoordsToVirtualCoords(
            ivec2(physicalPageCoords),
            ivec2(numPagesXY - 1),
            vsmClipMapIndex
        );

        // If this physical page is within the virtual bounds that the CPU wants to render
        // this frame, mark it as rendered instead of cleared
        if (virtualPageCoords.x >= startXY.x && virtualPageCoords.x <= endXY.x &&
            virtualPageCoords.y >= startXY.y && virtualPageCoords.y <= endXY.y) {

            if (updateCount < 3) {
                ++updateCount;
                currFramePageResidencyTable[physicalPageIndex].frameMarker = packFrameCountWithUpdateCount(frameCount, updateCount);
            }
            //else if (dirtyBit == VSM_PAGE_CLEARED_BIT) {
            else {
                updatedDirtyBit = VSM_PAGE_RENDERED_BIT;
                currFramePageResidencyTable[physicalPageIndex].info = packPageIdWithDirtyBit(pageId, updatedDirtyBit);
            }
        }
    }

    int pixelStepSize = int(gl_WorkGroupSize.x);

    if (dirtyBit == VSM_PAGE_DIRTY_BIT && frameMarker == frameCount) {
        if (gl_LocalInvocationID == 0) {
            uint newInfo = packPageIdWithDirtyBit(pageId, updatedDirtyBit);
            // currFramePageResidencyTable[physicalPageIndex].frameMarker = packFrameCountWithUpdateCount(frameCount, updateCount);
            currFramePageResidencyTable[physicalPageIndex].info = newInfo;
        }

        for (int x = vsmPixelStart.x + int(gl_LocalInvocationID.x); x < vsmPixelEnd.x; x += pixelStepSize) {
            for (int y = vsmPixelStart.y + int(gl_LocalInvocationID.y); y < vsmPixelEnd.y; y += pixelStepSize) { 
                clearPixel(ivec2(x, y));
            }
        }
    }
}