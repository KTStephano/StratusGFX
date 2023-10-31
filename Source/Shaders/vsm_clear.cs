// Clears physical pages to default depth value

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

uniform float clearValue = 1.0;
uniform int vsmClipMapIndex;
uniform uint numPagesXY;
uniform ivec2 startXY;
uniform ivec2 endXY;

shared ivec2 vsmSize;
shared ivec2 vsmMaxIndex;
shared ivec2 vsmPixelStart;
shared ivec2 vsmPixelEnd;
shared uint frameMarker;
shared uint residencyStatus;
shared uint memPool;
shared bool clearPage;
shared uint cascadeStepSize;

void clearPixel(in ivec2 physicalPixelCoords, in uint memPool) {
    imageStore(vsm, ivec3(physicalPixelCoords, int(memPool)), uvec4(floatBitsToUint(clearValue)));
}

void main() {
    ivec2 virtualPageCoords = ivec2(gl_WorkGroupID.xy);// + vec2(0.5);
    if (gl_LocalInvocationID == 0) {
        vsmSize = imageSize(vsm).xy;
        vsmMaxIndex = vsmSize - ivec2(1.0);
        clearPage = false;
        cascadeStepSize = uint(vsmClipMapIndex * numPagesXY * numPagesXY);

        vsmPixelStart = ivec2(
            int(virtualPageCoords.x) * VSM_MAX_NUM_TEXELS_PER_PAGE_XY,
            int(virtualPageCoords.y) * VSM_MAX_NUM_TEXELS_PER_PAGE_XY
        );

        vsmPixelEnd = vsmPixelStart + ivec2(VSM_MAX_NUM_TEXELS_PER_PAGE_XY);
    }

    barrier();

    //uint updatedDirtyBit = VSM_PAGE_CLEARED_BIT;

    if (gl_LocalInvocationID == 0) {
        uint virtualPageIndex = uint(virtualPageCoords.x + virtualPageCoords.y * numPagesXY + cascadeStepSize);
        uint updateMarker = pageGroupsToRender[virtualPageIndex];
        clearPage = updateMarker == VSM_VIRTUAL_SCREEN_UPDATE_MARKER;
        // If this physical page is within the virtual bounds that the CPU wants to render
        // this frame, mark it as rendered instead of cleared
        if (virtualPageCoords.x >= startXY.x && virtualPageCoords.x < endXY.x &&
            virtualPageCoords.y >= startXY.y && virtualPageCoords.y < endXY.y &&
            updateMarker > 0) {

            clearPage = true;//pageGroupsToRender[virtualPageIndex] > 1;
            --pageGroupsToRender[virtualPageIndex];
        }
    }

    int pixelStepSize = int(gl_WorkGroupSize.x);

    barrier();

    if (clearPage) {
        for (int x = vsmPixelStart.x + int(gl_LocalInvocationID.x); x < vsmPixelEnd.x; x += pixelStepSize) {
            for (int y = vsmPixelStart.y + int(gl_LocalInvocationID.y); y < vsmPixelEnd.y; y += pixelStepSize) {
                ivec2 localPixelCoords = ivec2(x, y);

                vec2 virtualUvCoords = convertLocalCoordsToVirtualUvCoords(
                    vec2(localPixelCoords),
                    vec2(vsmSize),
                    vsmClipMapIndex
                );

                ivec3 physicalPixelCoords = ivec3(floor(vsmConvertVirtualUVToPhysicalPixelCoords(
                    virtualUvCoords,
                    vec2(vsmSize),
                    numPagesXY,
                    vsmClipMapIndex
                )));

                if (physicalPixelCoords.z >= 0) {
                    clearPixel(physicalPixelCoords.xy, uint(physicalPixelCoords.z));
                }
            }
        }
    }
}