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

layout (std430, binding = VSM_PAGE_BOUNDING_BOX_BINDING_POINT) readonly buffer block6 {
    ClipMapBoundingBox clipMapBoundingBoxes[];
};

layout (std430, binding = VSM_NUM_PAGES_FREE_BINDING_POINT) coherent buffer block8 {
    int numPagesFree;
};

layout (std430, binding = VSM_PAGES_FREE_LIST_BINDING_POINT) readonly buffer block9 {
    uint pagesFreeList[];
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
// shared bool dirtyDueToClipOriginShift;

// shared ivec2 vsmSize;
// shared ivec2 vsmMaxIndex;
// shared bool pageDirty;
// shared bool performBoundsUpdate;
// // shared int localMinPageX;
// // shared int localMinPageY;
// // shared int localMaxPageX;
// // shared int localMaxPageY;
// shared uint cascadeStepSize;

// void clearPixel(in ivec2 physicalPixelCoords, in uint memPool) {
//     //imageStore(vsm, ivec3(physicalPixelCoords, vsmClipMapIndex), uvec4(clearValueBits));
//     imageStore(vsm, ivec3(physicalPixelCoords, int(memPool)), uvec4(floatBitsToUint(clearValue)));
// }

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
        //     cascadeNdc = vsmConvertClip0ToClipN(ndcClipOriginChange, cascade);
        //     virtualPixelChange = cascadeNdc * (float(vsmSize));

        //     // Determine virtual page bounds indicating which virtual pages have
        //     // changed since the last frame due to the clip origin being shifted
        //     if (virtualPixelChange.x < 0) {
        //         virtualPixelXYUpperBound.x = float(numPagesXY) + 1.0;
        //         virtualPixelXYLowerBound.x = floor(float(numPagesXY) + virtualPixelChange.x) - 5.0;
        //     }
        //     else {
        //         virtualPixelXYLowerBound.x = 0.0;
        //         virtualPixelXYUpperBound.x = ceil(virtualPixelChange.x) + 5.0;
        //     }

        //     if (virtualPixelChange.y < 0) {
        //         virtualPixelXYUpperBound.y = float(numPagesXY) + 1.0;
        //         virtualPixelXYLowerBound.y = floor(float(numPagesXY) + virtualPixelChange.y) - 5.0;
        //     }
        //     else {
        //         virtualPixelXYLowerBound.y = 0.0;
        //         virtualPixelXYUpperBound.y = ceil(virtualPixelChange.y) + 5.0;
        //     }

        //     // if (pageChange.x != 0.0) {
        //     //     virtualPixelXYLowerBound.x = 0.0;
        //     //     virtualPixelXYUpperBound.x = float(numPagesXY);
        //     // }

        //     // if (pageChange.y != 0.0) {
        //     //     virtualPixelXYLowerBound.y = 0.0;
        //     //     virtualPixelXYUpperBound.y = float(numPagesXY);
        //     // }

        //     // dirtyDueToClipOriginShift = 
        //     //     float(localPageCoords.x) >= virtualPixelXYLowerBound.x &&
        //     //     float(localPageCoords.x) <  virtualPixelXYUpperBound.x &&
        //     //     float(localPageCoords.y) >= virtualPixelXYLowerBound.y &&
        //     //     float(localPageCoords.y) <  virtualPixelXYUpperBound.y;
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
            vec2 ndcChange = ndc - cascadeNdcClipOriginChange;

            vec2 virtualUvCoords = convertLocalCoordsToVirtualUvCoords(
                vec2(localPixelCoords),
                vec2(vsmSize),
                cascade
            );

            // vec2 ndc = 2.0 * virtualUvCoords - 1.0;
            // vec2 prevNdc = ndc + cascadeNdcClipOriginChange;
            // vec2 prevVirtualUvCoords = wrapUVCoords(
            //     prevNdc * 0.5 + 0.5
            // );

            vec2 virtualPixelCoords = wrapIndex(virtualUvCoords, vec2(vsmSize));

            // ivec3 physicalPixelCoords = ivec3(floor(vsmConvertVirtualUVToPhysicalPixelCoords(
            //     virtualUvCoords,
            //     vec2(vsmSize),
            //     numPagesXY,
            //     cascade
            // )));

            ivec2 physicalPageCoords = ivec2(wrapIndex(virtualUvCoords, vec2(numPagesXY)));
            // ivec2 prevPhysicalPageCoords = ivec2(floor(prevVirtualUvCoords * vec2(numPagesXY)));
            uint physicalPageIndex = uint(physicalPageCoords.x + physicalPageCoords.y * numPagesXY + cascade * cascadeStepSize);
            uint unused;
            uint frameMarker;
            uint dirtyBit;

            unpackPageMarkerData(
                currFramePageResidencyTable[physicalPageIndex].frameMarker,
                frameMarker,
                unused,
                unused,
                unused,
                unused
            );

            unpackPageIdAndDirtyBit(
                currFramePageResidencyTable[physicalPageIndex].info,
                unused,
                dirtyBit
            );

            if (frameMarker > 0) {
                performBoundsUpdate = true;
                // atomicMin(localMinPageX, localPageCoords.x - 1);
                // atomicMin(localMinPageY, localPageCoords.y - 1);

                // atomicMax(localMaxPageX, localPageCoords.x + 1);
                // atomicMax(localMaxPageY, localPageCoords.y + 1);

                // If moving this pixel to previous NDC goes out of the [-1, 1] range, we migrated
                // to a different page and need to be regenerated
                if (dirtyBit > 0 || ndcChange.x <= -1 || ndcChange.x >= 1 || ndcChange.y <= -1 || ndcChange.y >= 1) {
                    pageDirty = true;
                }
            }
        }

        uint screenTileIndex = uint(localPageCoords.x + localPageCoords.y * numPagesXY + cascade * cascadeStepSize);
        uint updated = (performBoundsUpdate == false) ? 0 : (pageDirty ? 1 : pageGroupsToRender[screenTileIndex]);
        pageGroupsToRender[screenTileIndex] = updated;

        if (performBoundsUpdate) {
            atomicMin(clipMapBoundingBoxes[cascade].minPageX, localPageCoords.x - 1);
            atomicMin(clipMapBoundingBoxes[cascade].minPageY, localPageCoords.y - 1);

            atomicMax(clipMapBoundingBoxes[cascade].maxPageX, localPageCoords.x + 1);
            atomicMax(clipMapBoundingBoxes[cascade].maxPageY, localPageCoords.y + 1);
        }
    }
}