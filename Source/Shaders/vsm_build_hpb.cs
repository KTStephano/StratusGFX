// Loops over each cascade and builds a hierarchical page buffer (hpb)
// which is used for culling

STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

precision highp float;
precision highp int;
precision highp uimage2D;
precision highp sampler2D;
precision highp sampler2DArrayShadow;

// All the different levels of the hierarchical page buffer
layout (r32ui) readonly uniform uimage2DArray hpb0;
layout (r32ui) coherent uniform uimage2DArray hpb1;
layout (r32ui) coherent uniform uimage2DArray hpb2;
layout (r32ui) coherent uniform uimage2DArray hpb3;
layout (r32ui) coherent uniform uimage2DArray hpb4;
layout (r32ui) coherent uniform uimage2DArray hpb5;
layout (r32ui) coherent uniform uimage2DArray hpb6;
layout (r32ui) coherent uniform uimage2DArray hpb7;

// Max 8 (128x128 page table)
uniform int numMipLevels;

#define HPB_MIP_LEVEL(m) hpb##m
#define HPB_LOAD(m, xyz) imageLoad(HPB_MIP_LEVEL(m), xyz).r
#define HPB_STORE(m, xyz, v) imageStore(HPB_MIP_LEVEL(m), xyz, uvec4(v)); break

shared ivec2 widthHeight;
shared int cascadeIndex;

uint loadHpbValue(in int mipLevel, in ivec2 index) {
    ivec3 indexCascade = ivec3(index, cascadeIndex);
    switch (mipLevel) {
        case 0: return HPB_LOAD(0, indexCascade);
        case 1: return HPB_LOAD(1, indexCascade);
        case 2: return HPB_LOAD(2, indexCascade);
        case 3: return HPB_LOAD(3, indexCascade);
        case 4: return HPB_LOAD(4, indexCascade);
        case 5: return HPB_LOAD(5, indexCascade);
        case 6: return HPB_LOAD(6, indexCascade);
        case 7: return HPB_LOAD(7, indexCascade);
        //case 8: return HPB_LOAD(8, indexCascade);
        //case 9: return HPB_LOAD(9, indexCascade);
    }

    return uint(-1);
}

void storeHpbValue(in int mipLevel, in ivec2 index, in uint value) {
    ivec3 indexCascade = ivec3(index, cascadeIndex);
    switch (mipLevel) {
        //case 0: HPB_STORE(0, indexCascade, value);
        case 1: HPB_STORE(1, indexCascade, value);
        case 2: HPB_STORE(2, indexCascade, value);
        case 3: HPB_STORE(3, indexCascade, value);
        case 4: HPB_STORE(4, indexCascade, value);
        case 5: HPB_STORE(5, indexCascade, value);
        case 6: HPB_STORE(6, indexCascade, value);
        case 7: HPB_STORE(7, indexCascade, value);
        //case 8: HPB_STORE(8, indexCascade, value);
        //case 9: HPB_STORE(9, indexCascade, value);
    }
}

void main() {
    if (gl_LocalInvocationID == 0) {
        widthHeight  = ivec2(imageSize(HPB_MIP_LEVEL(0)).xy);
        cascadeIndex = int(gl_WorkGroupID.x);
    }

    barrier();

    int startX = int(2 * gl_LocalInvocationID.x);
    int startY = int(2 * gl_LocalInvocationID.y);
    int stride = int(2 * gl_WorkGroupSize.x);

    for (int mipLevel = 0; mipLevel < (numMipLevels - 1); ++mipLevel) {
        for (int x = startX; x < widthHeight.x; x += stride) {
            for (int y = startY; y < widthHeight.y; y += stride) {
                // Load neighboring pixels and merge them (max)

                // uint merged = loadHpbValue(mipLevel, ivec2(x, y));
                // for (int vx = -1; vx <= 1; ++vx) {
                //     for (int vy = -1; vy <= 1; ++vy) {

                //         if (vx == 0 && vy == 0) continue;
                //         uint v = loadHpbValue(mipLevel, ivec2(x + vx, y + vy));
                //         merged = max(merged, v);
                //     }
                // }

                uint v0 = loadHpbValue(mipLevel, ivec2(x     , y    ));
                uint v1 = loadHpbValue(mipLevel, ivec2(x + 1 , y    ));
                uint v2 = loadHpbValue(mipLevel, ivec2(x     , y + 1));
                uint v3 = loadHpbValue(mipLevel, ivec2(x + 1 , y + 1));

                uint merged = max(max(v0, v1), max(v2, v3));

                storeHpbValue(mipLevel + 1, ivec2(x / 2, y / 2), merged);
            }
        }

        // Wait for current mip level to finish
        barrier();

        if (gl_LocalInvocationID == 0) {
            widthHeight /= 2;
        }

        barrier();
    }
}