STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

precision highp float;
precision highp int;
precision highp uimage2D;
precision highp sampler2D;
precision highp sampler2DArrayShadow;

#include "vsm_common.glsl"

layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

uniform uint frameCount;
uniform uint numPagesXY;
uniform uint sunChanged; // Either 1 or 0

layout (std430, binding = 0) buffer block1 {
    int numPagesToFree;
};

layout (std430, binding = 1) buffer block2 {
    int pageIndices[];
};

layout (std430, binding = 2) buffer block4 {
    int renderPageIndices[];
};

layout (std430, binding = 3) buffer block3 {
    PageResidencyEntry prevFramePageResidencyTable[];
};

layout (std430, binding = 4) buffer block5 {
    PageResidencyEntry currFramePageResidencyTable[];
};

shared ivec2 residencyTableSize;

void main() {
    // if (gl_LocalInvocationID == 0) {
    //     residencyTableSize = imageSize(currFramePageResidencyTable).xy;
    // }

    // barrier();

    int tileXIndex = int(gl_GlobalInvocationID.x);
    int tileYIndex = int(gl_GlobalInvocationID.y);

    ivec2 tileCoords = ivec2(tileXIndex, tileYIndex);
    uint tileIndex = uint(tileCoords.x + tileCoords.y * int(numPagesXY));

    //uint prev = uint(imageLoad(prevFramePageResidencyTable, tileCoords).r);
    //uint current = uint(imageLoad(currFramePageResidencyTable, tileCoords).r);

    PageResidencyEntry prev = prevFramePageResidencyTable[tileIndex];
    PageResidencyEntry current = currFramePageResidencyTable[tileIndex];

    if (prev.frameMarker > 0 && current.frameMarker != frameCount) {
        //imageAtomicExchange(currFramePageResidencyTable, tileCoords, prev);
        //imageAtomicOr(currFramePageResidencyTable, tileCoords, prev);
        //prev.info = prev.info | sunChanged;// & VSM_PAGE_ID_MASK; // Get rid of the dirty bit
        current = prev;
        currFramePageResidencyTable[tileIndex] = prev;
    }

    if (current.frameMarker > 0) {
        if ((frameCount - current.frameMarker) > 30) {
            int original = atomicAdd(numPagesToFree, 1);
            pageIndices[2 * original] = tileCoords.x;
            pageIndices[2 * original + 1] = tileCoords.y;

            PageResidencyEntry markedNonResident;
            markedNonResident.frameMarker = 0;
            markedNonResident.info = 0;

            prevFramePageResidencyTable[tileIndex] = markedNonResident;
            currFramePageResidencyTable[tileIndex] = markedNonResident;
        }
        else if (sunChanged > 0) {
            current.info = current.info | 1;
            prevFramePageResidencyTable[tileIndex] = current;
            currFramePageResidencyTable[tileIndex] = current;
        }
    }
}