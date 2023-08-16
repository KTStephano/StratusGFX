STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout (r32i) readonly uniform iimage2D prevFramePageResidencyTable;
layout (r32i) readonly uniform iimage2D currFramePageResidencyTable;

layout (std430, binding = 0) buffer block1 {
    int numPagesToFree;
};

layout (std430, binding = 1) buffer block2 {
    int pageIndices[];
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

    int prev = int(imageLoad(prevFramePageResidencyTable, tileCoords).r);
    int current = int(imageLoad(currFramePageResidencyTable, tileCoords).r);

    if (prev == 1 && current == 0) {
        int original = atomicAdd(numPagesToFree, 1);
        pageIndices[2 * original] = tileCoords.x;
        pageIndices[2 * original + 1] = tileCoords.y;
    }
}