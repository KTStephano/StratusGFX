STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

precision highp float;
precision highp int;
precision highp uimage2D;

#include "vsm_common.glsl"

layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

uniform uint frameCount;
uniform uint numPagesXY;

layout (std430, binding = 0) buffer block1 {
    int numPagesToFree;
};

layout (std430, binding = 1) buffer block2 {
    int pageIndices[];
};

layout (std430, binding = 2) buffer block4 {
    int renderPageIndices[];
};

layout (std430, binding = 3) readonly buffer block3 {
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

    uint prev = prevFramePageResidencyTable[tileIndex].frameMarker;
    uint current = currFramePageResidencyTable[tileIndex].frameMarker;

    if (prev > 0 && current == 0) {
        //imageAtomicExchange(currFramePageResidencyTable, tileCoords, prev);
        //imageAtomicOr(currFramePageResidencyTable, tileCoords, prev);
        atomicExchange(currFramePageResidencyTable[tileIndex].frameMarker, prev);

        // Don't want to render tiles from last frame that aren't visible
        //renderPageIndices[tileCoords.x + tileCoords.y * residencyTableSize.x] = 0;
    }

    if (current > 0 && (frameCount - current) > 60) {
        int original = atomicAdd(numPagesToFree, 1);
        pageIndices[2 * original] = tileCoords.x;
        pageIndices[2 * original + 1] = tileCoords.y;

        //imageAtomicExchange(prevFramePageResidencyTable, tileCoords, 0);
        //imageAtomicExchange(currFramePageResidencyTable, tileCoords, 0);

        atomicExchange(prevFramePageResidencyTable[tileIndex].frameMarker, 0);
        atomicExchange(currFramePageResidencyTable[tileIndex].frameMarker, 0);

        //renderPageIndices[tileCoords.x + tileCoords.y * residencyTableSize.x] = 0;
    }
}