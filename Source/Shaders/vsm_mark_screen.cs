// Full screen pass over the virtual screen (per cascade) which determines
// screen X/Y bounds and checks for regions that are dirty due to origin shifts

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
layout (r32ui) uniform uimage2DArray hpb;

layout (std430, binding = VSM_PAGE_BOUNDING_BOX_BINDING_POINT) readonly buffer block6 {
    ClipMapBoundingBox clipMapBoundingBoxes[];
};

uniform float clearValue = 1.0;
uniform uint numPagesXY;
// Measures change (in NDC units) of the clip origin since last frame
uniform vec2 ndcClipOriginChange;

shared ivec2 vsmSize;
shared ivec2 vsmMaxIndex;
shared uint cascadeStepSize;
shared vec2 cascadeNdc;
shared vec2 virtualPixelChange;
shared vec2 virtualPixelXYLowerBound;
shared vec2 virtualPixelXYUpperBound;
shared vec2 cascadeNdcClipOriginChange;

void main() {
    if (gl_LocalInvocationID == 0) {
        vsmSize = imageSize(vsm).xy;
        vsmMaxIndex = vsmSize - ivec2(1);
        cascadeStepSize = numPagesXY * numPagesXY;
    }

    barrier();

    for (int cascade = 0; cascade < int(vsmNumCascades); ++cascade) {
        bool performBoundsUpdate = false;
        bool pageDirty = false;
        ivec2 localPageCoords = ivec2(gl_GlobalInvocationID.xy);// + vec2(0.5);

        if (gl_LocalInvocationID == 0) {
            cascadeNdcClipOriginChange = vsmConvertClip0ToClipN(ndcClipOriginChange, cascade);
        }

        barrier();

        ivec2 localPixelCoordsStart = localPageCoords * ivec2(VSM_MAX_NUM_TEXELS_PER_PAGE_XY);
        ivec2 localPixelCoordsEnd = localPixelCoordsStart + ivec2(VSM_MAX_NUM_TEXELS_PER_PAGE_XY - 1);

        // Compute 4 local corners
        ivec2 localPixelCorner1 = ivec2(localPixelCoordsStart.x, localPixelCoordsStart.y);
        ivec2 localPixelCorner2 = ivec2(localPixelCoordsStart.x, localPixelCoordsEnd.y);
        ivec2 localPixelCorner3 = ivec2(localPixelCoordsEnd.x, localPixelCoordsStart.y);
        ivec2 localPixelCorner4 = ivec2(localPixelCoordsEnd.x, localPixelCoordsEnd.y);

        ivec2 corners[4] = ivec2[4](
            localPixelCorner1,
            localPixelCorner2,
            localPixelCorner3,
            localPixelCorner4
        );

        for (int i = 0; i < 4 && !(performBoundsUpdate && pageDirty); ++i) {
            ivec2 localPixelCoords = ivec2(corners[i].x, corners[i].y);

            vec2 ndc = vec2(2 * localPixelCoords) / vec2(vsmSize) - 1.0;
            // Apply motion vector to local ndc
            vec2 ndcChange = ndc - cascadeNdcClipOriginChange;

            vec2 virtualUvCoords = convertLocalCoordsToVirtualUvCoords(
                vec2(localPixelCoords),
                vec2(vsmSize),
                cascade
            );

            vec2 virtualPixelCoords = wrapIndex(virtualUvCoords, vec2(vsmSize));

            ivec2 physicalPageCoords = ivec2(wrapIndex(virtualUvCoords, vec2(numPagesXY)));
            // ivec2 prevPhysicalPageCoords = ivec2(floor(prevVirtualUvCoords * vec2(numPagesXY)));
            uint physicalPageIndex = uint(physicalPageCoords.x + physicalPageCoords.y * numPagesXY + cascade * cascadeStepSize);

            uint frameMarker = unpackFrameMarker(
                currFramePageResidencyTable[physicalPageIndex].info
            );

            uint dirtyBit = unpackDirtyBit(
                currFramePageResidencyTable[physicalPageIndex].info
            );

            if (frameMarker > 0) {
                performBoundsUpdate = true;
                pageDirty = true;

                // If moving this pixel to previous NDC goes out of the [-1, 1] range, it was not visible last
                // frame before the origin shift and will be wrapped around to the other side
                if (dirtyBit > 0 || ndcChange.x < -1 || ndcChange.x > 1 || ndcChange.y < -1 || ndcChange.y > 1) {
                    pageDirty = true;
                }
            }
        }

        uint screenTileIndex = uint(localPageCoords.x + localPageCoords.y * numPagesXY + cascade * cascadeStepSize);
        uint updated = (performBoundsUpdate == false) ? 0 : (pageDirty ? 2 : pageGroupsToRender[screenTileIndex]);
        pageGroupsToRender[screenTileIndex] = updated;

        imageStore(hpb, ivec3(localPageCoords.xy, cascade), uvec4(floatBitsToUint(updated > 0 ? 1024.0 : 0.0)));

        if (performBoundsUpdate) {
            atomicMin(clipMapBoundingBoxes[cascade].minPageX, localPageCoords.x - 1);
            atomicMin(clipMapBoundingBoxes[cascade].minPageY, localPageCoords.y - 1);

            atomicMax(clipMapBoundingBoxes[cascade].maxPageX, localPageCoords.x + 1);
            atomicMax(clipMapBoundingBoxes[cascade].maxPageY, localPageCoords.y + 1);
        }
    }
}