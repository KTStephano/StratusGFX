STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

precision highp float;
precision highp int;
precision highp uimage2D;
precision highp sampler2D;
precision highp sampler2DArrayShadow;

// #include "vsm_common.glsl"
#include "bindings.glsl"

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

uniform uint numPagesXY;

layout (std430, binding = VSM_NUM_PAGES_TO_UPDATE_BINDING_POINT) readonly buffer block1 {
    int numPagesToUpdate;
};

layout (std430, binding = VSM_PAGE_INDICES_BINDING_POINT) readonly buffer block2 {
    int pageIndices[];
};

layout (std430, binding = VSM_NUM_PAGES_FREE_BINDING_POINT) buffer block8 {
    int numPagesFree;
};

layout (std430, binding = VSM_PAGES_FREE_LIST_BINDING_POINT) buffer block9 {
    uint pagesFreeList[];
};

void main() {
    uint stepSize = gl_WorkGroupSize.x;

    if (gl_LocalInvocationID == 0) {
        int maxNumPages = int(numPagesXY * numPagesXY);
        if (numPagesFree > maxNumPages) {
            numPagesFree = maxNumPages;
        }
    }

    barrier();

    for (uint i = gl_LocalInvocationIndex; i < numPagesToUpdate; i += stepSize) {
        int x = pageIndices[3 * i + 1];
        int y = pageIndices[3 * i + 2];

        if (x < 0 || y < 0) {
            x = abs(x) - 1;
            y = abs(y) - 1;

            int nextPage = atomicAdd(numPagesFree, -1) - 1;
            pagesFreeList[2 * nextPage] = uint(x);
            pagesFreeList[2 * nextPage + 1] = uint(y);
        }
    }
}