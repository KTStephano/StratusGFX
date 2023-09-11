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

layout (std430, binding = VSM_NUM_PAGES_TO_UPDATE_BINDING_POINT) coherent buffer block1 {
    int numPagesToUpdate;
};

layout (std430, binding = VSM_PAGE_INDICES_BINDING_POINT) coherent buffer block2 {
    int pageIndices[];
};

// layout (std430, binding = 2) buffer block4 {
//     int renderPageIndices[];
// };

// layout (std430, binding = VSM_PREV_FRAME_RESIDENCY_TABLE_BINDING) coherent buffer block3 {
//     PageResidencyEntry prevFramePageResidencyTable[];
// };

layout (std430, binding = VSM_PAGE_BOUNDING_BOX_BINDING_POINT) readonly buffer block6 {
    ClipMapBoundingBox clipMapBoundingBoxes[];
};

layout (std430, binding = VSM_PAGE_GROUPS_TO_RENDER_BINDING_POINT) coherent buffer block7 {
    uint pageGroupsToRender[];
};

layout (std430, binding = VSM_NUM_PAGES_FREE_BINDING_POINT) coherent buffer block8 {
    int numPagesFree;
};

layout (std430, binding = VSM_PAGES_FREE_LIST_BINDING_POINT) readonly buffer block9 {
    uint pagesFreeList[];
};

shared int localMinPageX;
shared int localMinPageY;
shared int localMaxPageX;
shared int localMaxPageY;
shared int cascadeStepSize;

// bool requestPageAlloc(in ivec2 physicalPage, out uint physicalPageX, out uint physicalPageY, in uint memPool) {
//     // int maxPage = int(numPagesXY * numPagesXY);
//     // int nextPage = atomicAdd(numPagesFree, 1);
//     // if (nextPage >= maxPage) {
//     //     physicalPageX = 0;
//     //     physicalPageY = 0;
//     //     return false;
//     // }

//     // uint px = pagesFreeList[2 * nextPage];
//     // uint py = pagesFreeList[2 * nextPage + 1];
//     uint px = uint(physicalPage.x);
//     uint py = uint(physicalPage.y);
//     physicalPageX = px;
//     physicalPageY = py;

//     int original = atomicAdd(numPagesToUpdate, 1);
//     pageIndices[3 * original] = int(memPool);
//     pageIndices[3 * original + 1] = int(px) + 1;
//     pageIndices[3 * original + 2] = int(py) + 1;

//     return true;
// }

// void requestPageDealloc(in ivec2 pageCoords, in uint memPool) {
//     // int maxPage = int(numPagesXY * numPagesXY);
//     // int nextPage = atomicAdd(numPagesFree, -1);
//     // while (nextPage > maxPage) {
//     //     nextPage = atomicAdd(numPagesFree, -1);
//     // }

//     // nextPage = nextPage - 1;
//     // pagesFreeList[2 * nextPage] = uint(pageCoords.x);
//     // pagesFreeList[2 * nextPage + 1] = uint(pageCoords.y);

//     int original = atomicAdd(numPagesToUpdate, 1);
//     pageIndices[3 * original] = int(memPool);
//     pageIndices[3 * original + 1] = -(pageCoords.x + 1);
//     pageIndices[3 * original + 2] = -(pageCoords.y + 1);
// }

bool requestPageAlloc(in ivec2 physicalPage, out uint physicalPageX, out uint physicalPageY, out uint memPool) {
    int maxPage = int(vsmNumCascades * numPagesXY * numPagesXY);
    int nextPage = atomicAdd(numPagesFree, 1);
    if (nextPage >= maxPage) {
        physicalPageX = 0;
        physicalPageY = 0;
        memPool = 0;
        return false;
    }

    uint mem = pagesFreeList[3 * nextPage];
    uint px  = pagesFreeList[3 * nextPage + 1];
    uint py  = pagesFreeList[3 * nextPage + 2];
    physicalPageX = px;
    physicalPageY = py;
    memPool = mem;

    int original = atomicAdd(numPagesToUpdate, 1);
    pageIndices[3 * original]     = int(mem);
    pageIndices[3 * original + 1] = int(px) + 1;
    pageIndices[3 * original + 2] = int(py) + 1;

    return true;
}

void requestPageDealloc(in ivec2 pageCoords, in uint memPool) {
    // int maxPage = int(numPagesXY * numPagesXY);
    // int nextPage = atomicAdd(numPagesFree, -1);
    // while (nextPage > maxPage) {
    //     nextPage = atomicAdd(numPagesFree, -1);
    // }

    // nextPage = nextPage - 1;
    // pagesFreeList[2 * nextPage] = uint(pageCoords.x);
    // pagesFreeList[2 * nextPage + 1] = uint(pageCoords.y);

    int original = atomicAdd(numPagesToUpdate, 1);
    pageIndices[3 * original] = int(memPool);
    pageIndices[3 * original + 1] = -(pageCoords.x + 1);
    pageIndices[3 * original + 2] = -(pageCoords.y + 1);
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

        ivec2 virtualPageCoords = ivec2(tileXIndex, tileYIndex);

        ivec2 physicalPageCoords = ivec2(floor(convertVirtualCoordsToPhysicalCoords(
            vec2(virtualPageCoords) + vec2(0.5),
            vec2(int(numPagesXY) - 1),
            cascade
        )));

        uint pageIndex = uint(physicalPageCoords.x + physicalPageCoords.y * int(numPagesXY) + cascade * cascadeStepSize);

        //uint prev = uint(imageLoad(prevFramePageResidencyTable, virtualPageCoords).r);
        //uint current = uint(imageLoad(currFramePageResidencyTable, virtualPageCoords).r);

        PageResidencyEntry current = currFramePageResidencyTable[pageIndex];

        uint frameMarker;
        uint physicalPageX;
        uint physicalPageY;
        uint memPool;
        uint pageResident;
        unpackPageMarkerData(
            current.frameMarker,
            frameMarker,
            physicalPageX,
            physicalPageY,
            memPool,
            pageResident
        );

        uint pageId;
        uint dirtyBit;
        unpackPageIdAndDirtyBit(current.info, pageId, dirtyBit);

        // if (prevPageId == pageId && prevDirtyBit > 0 && current.frameMarker == frameCount) {
        //     dirtyBit = 1;
        //     current.info |= dirtyBit;
        // }

        // Take the physical coords and convert them to virtual coords for the current frame
        // ivec2 physicalPageCoords = ivec2(floor(convertVirtualCoordsToPhysicalCoords(
        //     vec2(virtualPageCoords) + vec2(0.5),
        //     vec2(int(numPagesXY) - 1),
        //     cascade
        // )));

        //uint virtualPageIndex = uint(virtualPageCoords.x + virtualPageCoords.y * int(numPagesXY) + cascade * cascadeStepSize);

        if (frameMarker > 0) {
            // Frame has not been needed for more than 30 frames and needs to be freed
            if (frameMarker > 1) {
                if (pageResident > 0) {
                    dirtyBit = 0;
                    requestPageDealloc(ivec2(int(physicalPageX), int(physicalPageY)), memPool);

                    PageResidencyEntry markedNonResident;
                    markedNonResident.frameMarker = 0;
                    markedNonResident.info = 0;
                    frameMarker = 0;

                    currFramePageResidencyTable[pageIndex] = markedNonResident;

                    current = markedNonResident;
                }
            }
            else {
                //uint newPageResidencyStatus = 2; // 2 means nothing needs to be done
                uint newPageResidencyStatus = 2;
                // Page was requested this frame but is not currently resident
                if (pageResident == 0) {
                    //memPool = uint(cascade);
                    if (requestPageAlloc(physicalPageCoords, physicalPageX, physicalPageY, memPool) == false) {
                        // Failed to allocate
                        newPageResidencyStatus = 0;
                    }
                    else {
                        newPageResidencyStatus = 1;
                    }
                    dirtyBit = VSM_PAGE_DIRTY_BIT;
                    current.info = (current.info & VSM_PAGE_ID_MASK) | VSM_PAGE_DIRTY_BIT;
                }
                else if (sunChanged > 0) {
                    dirtyBit = VSM_PAGE_DIRTY_BIT;
                    current.info = (current.info & VSM_PAGE_ID_MASK) | VSM_PAGE_DIRTY_BIT;
                }
                else if (dirtyBit == VSM_PAGE_RENDERED_BIT && pageResident >= 2) { //>= VSM_MAX_NUM_TEXELS_PER_PAGE) {
                    dirtyBit = 0;
                    current.info = current.info & VSM_PAGE_ID_MASK;
                }

                //current.info = current.info & VSM_PAGE_ID_MASK;

                current.frameMarker = packPageMarkerData(
                    2,//frameMarker + 1, 
                    physicalPageX,
                    physicalPageY,
                    memPool,
                    newPageResidencyStatus
                );

                currFramePageResidencyTable[pageIndex] = current;

                // prev = current;
                // prev.info &= VSM_PAGE_ID_MASK;
                // prevFramePageResidencyTable[pageIndex] = prev;
            }
        }

        // Do a final check so that we can determine if this page needs
        // to be processed by culling and later rendering
        // unpackPageIdAndDirtyBit(current.info, pageId, dirtyBit);

        uint pageGroupMarker = 0;

        //if (dirtyBit > 0 && frameMarker == frameCount) {
        if (frameMarker > 0) {
        //if (dirtyBit > 0) {
        //if (frameMarker > 0) {
            //pageGroupMarker = frameCount;

            atomicMin(localMinPageX, virtualPageCoords.x - 1);
            atomicMin(localMinPageY, virtualPageCoords.y - 1);

            atomicMax(localMaxPageX, virtualPageCoords.x + 1);
            atomicMax(localMaxPageY, virtualPageCoords.y + 1);
        }

        if (dirtyBit > 0) {
            pageGroupMarker = 1;
        }

        uint virtualPageIndex = uint(virtualPageCoords.x + virtualPageCoords.y * int(numPagesXY) + cascade * cascadeStepSize);
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