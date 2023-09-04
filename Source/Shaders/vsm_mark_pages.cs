STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

precision highp float;
precision highp int;
precision highp uimage2D;
precision highp sampler2D;
precision highp sampler2DArrayShadow;

#include "vsm_common.glsl"
#include "bindings.glsl"

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

uniform uint frameCount;
uniform uint numPagesXY;
uniform uint sunChanged; // Either 1 or 0

layout (std430, binding = VSM_NUM_PAGES_TO_UPDATE_BINDING_POINT) buffer block1 {
    int numPagesToUpdate;
};

layout (std430, binding = VSM_PAGE_INDICES_BINDING_POINT) buffer block2 {
    int pageIndices[];
};

// layout (std430, binding = 2) buffer block4 {
//     int renderPageIndices[];
// };

layout (std430, binding = VSM_PREV_FRAME_RESIDENCY_TABLE_BINDING) coherent buffer block3 {
    PageResidencyEntry prevFramePageResidencyTable[];
};

layout (std430, binding = VSM_CURR_FRAME_RESIDENCY_TABLE_BINDING) coherent buffer block5 {
    PageResidencyEntry currFramePageResidencyTable[];
};

layout (std430, binding = VSM_PAGE_BOUNDING_BOX_BINDING_POINT) buffer block6 {
    ClipMapBoundingBox clipMapBoundingBoxes[];
};

layout (std430, binding = VSM_PAGE_GROUPS_TO_RENDER_BINDING_POINT) buffer block7 {
    uint pageGroupsToRender[];
};

shared int localMinPageX;
shared int localMinPageY;
shared int localMaxPageX;
shared int localMaxPageY;
shared int cascadeStepSize;

void requestPageAlloc(in ivec2 tileCoords, in int cascade) {
    int original = atomicAdd(numPagesToUpdate, 1);
    pageIndices[3 * original] = cascade;
    pageIndices[3 * original + 1] = tileCoords.x + 1;
    pageIndices[3 * original + 2] = tileCoords.y + 1;
}

void requestPageDealloc(in ivec2 tileCoords, in int cascade) {
    int original = atomicAdd(numPagesToUpdate, 1);
    pageIndices[3 * original] = cascade;
    pageIndices[3 * original + 1] = -(tileCoords.x + 1);
    pageIndices[3 * original + 2] = -(tileCoords.y + 1);
}

void main() {
    if (gl_LocalInvocationID == 0) {
        cascadeStepSize = int(numPagesXY * numPagesXY);
    }

    barrier();

    for (int cascade = 0; cascade < int(vsmNumCascades); ++cascade) {
        if (gl_LocalInvocationID == 0) {
            localMinPageX = int(numPagesXY) + 1;
            localMinPageY = int(numPagesXY) + 1;
            localMaxPageX = -1;
            localMaxPageY = -1;

            // localMinPageX = 0;
            // localMinPageY = 0;
            // localMaxPageX = int(numPagesXY) - 1;
            // localMaxPageY = int(numPagesXY) - 1;
        }

        barrier();

        int tileXIndex = int(gl_GlobalInvocationID.x);
        int tileYIndex = int(gl_GlobalInvocationID.y);

        ivec2 tileCoords = ivec2(tileXIndex, tileYIndex);
        uint tileIndex = uint(tileCoords.x + tileCoords.y * int(numPagesXY) + cascade * cascadeStepSize);

        //uint prev = uint(imageLoad(prevFramePageResidencyTable, tileCoords).r);
        //uint current = uint(imageLoad(currFramePageResidencyTable, tileCoords).r);

        PageResidencyEntry prev = prevFramePageResidencyTable[tileIndex];
        PageResidencyEntry current = currFramePageResidencyTable[tileIndex];

        uint prevPageId;
        uint prevDirtyBit;
        unpackPageIdAndDirtyBit(prev.info, prevPageId, prevDirtyBit);

        if (prev.frameMarker > 0 && current.frameMarker != frameCount) {
            current = prev;
            currFramePageResidencyTable[tileIndex] = prev;
        }

        uint pageId;
        uint dirtyBit;
        unpackPageIdAndDirtyBit(current.info, pageId, dirtyBit);

        // if (prevPageId == pageId && prevDirtyBit > 0 && current.frameMarker == frameCount) {
        //     dirtyBit = 1;
        //     current.info |= dirtyBit;
        // }

        // Take the physical coords and convert them to virtual coords for the current frame
        ivec2 virtualPageCoords = ivec2(floor(convertPhysicalCoordsToVirtualCoords(
            tileCoords,
            ivec2(int(numPagesXY) - 1),
            cascade
        )));

        uint virtualPageIndex = uint(virtualPageCoords.x + virtualPageCoords.y * int(numPagesXY) + cascade * cascadeStepSize);

        if (current.frameMarker > 0) {
            // Frame has not been needed for more than 30 frames and needs to be freed
            if ((frameCount - current.frameMarker) > 30) {
                requestPageDealloc(tileCoords, cascade);

                PageResidencyEntry markedNonResident;
                markedNonResident.frameMarker = 0;
                markedNonResident.info = 0;

                prevFramePageResidencyTable[tileIndex] = markedNonResident;
                currFramePageResidencyTable[tileIndex] = markedNonResident;

                current = markedNonResident;
            }
            else {
                // Page was requested this frame but is not currently resident
                if (prev.frameMarker == 0) {
                    dirtyBit = 1;
                    current.info = (current.info & VSM_PAGE_ID_MASK) | 1;
                    requestPageAlloc(tileCoords, cascade); 
                }
                else if (sunChanged > 0) {
                    dirtyBit = 1;
                    current.info = (current.info & VSM_PAGE_ID_MASK) | 1;
                }
                else if (dirtyBit >= VSM_MAX_NUM_TEXELS_PER_PAGE) {
                    dirtyBit = 0;
                    current.info = current.info & VSM_PAGE_ID_MASK;
                }

                //current.info = current.info & VSM_PAGE_ID_MASK;

                // prevFramePageResidencyTable[tileIndex] = current;
                currFramePageResidencyTable[tileIndex] = current;

                prev = current;
                prev.info &= VSM_PAGE_ID_MASK;
                prevFramePageResidencyTable[tileIndex] = prev;
            }
        }

        // Do a final check so that we can determine if this page needs
        // to be processed by culling and later rendering
        // unpackPageIdAndDirtyBit(current.info, pageId, dirtyBit);

        uint pageGroupMarker = 0;

        if (dirtyBit > 0 && current.frameMarker == frameCount) {
            pageGroupMarker = frameCount;

            atomicMin(localMinPageX, virtualPageCoords.x);
            atomicMin(localMinPageY, virtualPageCoords.y);

            atomicMax(localMaxPageX, virtualPageCoords.x + 1);
            atomicMax(localMaxPageY, virtualPageCoords.y + 1);
        }

        pageGroupsToRender[virtualPageIndex] = pageGroupMarker;

        barrier();

        if (gl_LocalInvocationID == 0) {
            atomicMin(clipMapBoundingBoxes[cascade].minPageX, localMinPageX);
            atomicMin(clipMapBoundingBoxes[cascade].minPageY, localMinPageY);

            atomicMax(clipMapBoundingBoxes[cascade].maxPageX, localMaxPageX);
            atomicMax(clipMapBoundingBoxes[cascade].maxPageY, localMaxPageY);
        }

        barrier();
    }
}