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
shared bool clearPage;

void clearPixel(in ivec2 physicalPixelCoords) {
    imageStore(vsm, ivec3(physicalPixelCoords, vsmClipMapIndex), uvec4(clearValueBits));
}

void main() {
    if (gl_LocalInvocationID == 0) {
        clearValueBits = floatBitsToUint(clearValue);
        vsmSize = imageSize(vsm).xy;
        vsmMaxIndex = vsmSize - ivec2(1.0);
        clearPage = false;
    }

    barrier();

    vec2 virtualPageCoords = vec2(gl_WorkGroupID.xy) + vec2(0.5);

    ivec2 physicalPageCoords = ivec2(floor(convertVirtualCoordsToPhysicalCoords(
        vec2(virtualPageCoords),
        vec2(numPagesXY - ivec2(1)),
        vsmClipMapIndex
    )));

    uint physicalPageIndex = uint(physicalPageCoords.x + physicalPageCoords.y * numPagesXY.x + vsmClipMapIndex * numPagesXY.x * numPagesXY.y);

    if (gl_LocalInvocationID == 0) {
        vsmPixelStart = ivec2(physicalPageCoords.x * VSM_MAX_NUM_TEXELS_PER_PAGE_XY, physicalPageCoords.y * VSM_MAX_NUM_TEXELS_PER_PAGE_XY);
        vsmPixelEnd = vsmPixelStart + ivec2(VSM_MAX_NUM_TEXELS_PER_PAGE_XY);
        
        // unpackFrameCountAndUpdateCount(
        //     currFramePageResidencyTable[physicalPageIndex].frameMarker,
        //     frameMarker,
        //     updateCount
        // );
    }

    uint pageId;
    uint dirtyBit;
    unpackPageIdAndDirtyBit(
        currFramePageResidencyTable[physicalPageIndex].info, 
        pageId,
        dirtyBit
    );

    barrier();

    //uint updatedDirtyBit = VSM_PAGE_CLEARED_BIT;

    //if (gl_LocalInvocationID == 0 && dirtyBit > 0) {// && frameMarker == frameCount) {
    if (gl_LocalInvocationID == 0) {
        // vec2 virtualPageCoords = convertPhysicalCoordsToVirtualCoords(
        //     ivec2(physicalPageCoords),
        //     ivec2(numPagesXY - 1),
        //     vsmClipMapIndex
        // );

        // If this physical page is within the virtual bounds that the CPU wants to render
        // this frame, mark it as rendered instead of cleared
        if (virtualPageCoords.x >= startXY.x && virtualPageCoords.x <= endXY.x &&
            virtualPageCoords.y >= startXY.y && virtualPageCoords.y <= endXY.y) {
        //if (true) {

            clearPage = true;

            if (dirtyBit > 0) {
                //clearPage = true;
            //if (frameMarker > 0) {
                //updatedDirtyBit = VSM_PAGE_RENDERED_BIT;
                currFramePageResidencyTable[physicalPageIndex].info = packPageIdWithDirtyBit(pageId, VSM_PAGE_RENDERED_BIT);
            }
        }
    }

    int pixelStepSize = int(gl_WorkGroupSize.x);

    barrier();

    //if (dirtyBit == VSM_PAGE_DIRTY_BIT && frameMarker == frameCount) {
    if (clearPage) {
        for (int x = vsmPixelStart.x + int(gl_LocalInvocationID.x); x < vsmPixelEnd.x; x += pixelStepSize) {
            for (int y = vsmPixelStart.y + int(gl_LocalInvocationID.y); y < vsmPixelEnd.y; y += pixelStepSize) { 
                clearPixel(ivec2(x, y));
            }
        }
    }
}