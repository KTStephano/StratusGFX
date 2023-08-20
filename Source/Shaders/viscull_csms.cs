STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

#include "common.glsl"
#include "aabb.glsl"

uniform uint numDrawCalls;
uniform uint maxDrawCommands;
uniform mat4 cascadeViewProj[4];
uniform uint numCascades;
uniform uint numPageGroups;

layout (std430, binding = 2) readonly buffer inputBlock3 {
    mat4 modelTransforms[];
};

layout (std430, binding = 3) readonly buffer inputBlock4 {
    AABB aabbs[];
};

// Cascades 0 and 1
layout (std430, binding = 1) readonly buffer inputBlock1 {
    DrawElementsIndirectCommand cascade01DrawCalls[];
};

// Cascades 2 and 3
layout (std430, binding = 4) readonly buffer inputBlock2 {
    DrawElementsIndirectCommand cascade23DrawCalls[];
};

// Outputs for all 4 cascades
layout (std430, binding = 5) buffer outputBlock1 {
    DrawElementsIndirectCommand outDrawCallsCascade0[];
};

layout (std430, binding = 6) buffer outputBlock2 {
    DrawElementsIndirectCommand outDrawCallsCascade1[];
};

layout (std430, binding = 7) buffer outputBlock3 {
    DrawElementsIndirectCommand outDrawCallsCascade2[];
};

layout (std430, binding = 8) buffer outputBlock4 {
    DrawElementsIndirectCommand outDrawCallsCascade3[];
};

// For these we are only responsible for zeroing out the memory
layout (std430, binding = 9) buffer outputBlock5 {
    DrawElementsIndirectCommand outDrawCalls2Cascade0[];
};

layout (std430, binding = 10) buffer outputBlock6 {
    DrawElementsIndirectCommand outDrawCalls2Cascade1[];
};

layout (std430, binding = 11) buffer outputBlock7 {
    DrawElementsIndirectCommand outDrawCalls2Cascade2[];
};

layout (std430, binding = 12) buffer outputBlock8 {
    DrawElementsIndirectCommand outDrawCalls2Cascade3[];
};

shared vec4 cascadeFrustumPlanes0[6];
shared vec4 cascadeFrustumPlanes1[6];
shared vec4 cascadeFrustumPlanes2[6];
shared vec4 cascadeFrustumPlanes3[6];

void main() {
    // Defines local work group from layout local size tag above
    uint localWorkGroupSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y;

    // Extract world-space frustum planes
#define INITIALIZE_CASCADE_PLANES(index, planes)            \
    if (gl_LocalInvocationIndex == index && index < numCascades) { \
        mat4 vpt = transpose(cascadeViewProj[index]);       \
        planes[0] = vpt[3] + vpt[0];                        \
        planes[1] = vpt[3] - vpt[0];                        \
        planes[2] = vpt[3] + vpt[1];                        \
        planes[3] = vpt[3] - vpt[1];                        \
        planes[4] = vpt[3] + vpt[2];                        \
        planes[5] = vpt[3] - vpt[2];                        \
    }

    INITIALIZE_CASCADE_PLANES(0, cascadeFrustumPlanes0);
    INITIALIZE_CASCADE_PLANES(1, cascadeFrustumPlanes1);
    INITIALIZE_CASCADE_PLANES(2, cascadeFrustumPlanes2);
    INITIALIZE_CASCADE_PLANES(3, cascadeFrustumPlanes3);

    barrier();

#define PERFORM_VISCULL_FOR_CASCADE(index, planes, aabb, draw, out1, out2) \
    if (!isAabbVisible(planes, aabb)) {                             \
        draw.instanceCount = 0;                                     \
    } else {                                                        \
        draw.instanceCount = 1;                                     \
    }                                                               \
    out1[index] = draw;                                             \
    draw.instanceCount = 0;                                         \
    out2[index] = draw;

    // for (uint group = 0; group < numPageGroups; ++group) {          \
    //     out2[group * maxDrawCommands + index] = draw;               \
    // }
   
    for (uint i = gl_LocalInvocationIndex; i < numDrawCalls; i += localWorkGroupSize) {
        AABB aabb = transformAabb(aabbs[i], modelTransforms[i]);

        // Cascades 0, 1
        if (numCascades > 0) {
            DrawElementsIndirectCommand draw = cascade01DrawCalls[i];
            PERFORM_VISCULL_FOR_CASCADE(i, cascadeFrustumPlanes0, aabb, draw, outDrawCallsCascade0, outDrawCalls2Cascade0);
            if (numCascades > 1) {
                PERFORM_VISCULL_FOR_CASCADE(i, cascadeFrustumPlanes1, aabb, draw, outDrawCallsCascade1, outDrawCalls2Cascade1);
            }
        }

        // Cascades 2, 3
        if (numCascades > 2) {
            DrawElementsIndirectCommand draw = cascade23DrawCalls[i];
            PERFORM_VISCULL_FOR_CASCADE(i, cascadeFrustumPlanes2, aabb, draw, outDrawCallsCascade2, outDrawCalls2Cascade2);
            if (numCascades > 3) {
                PERFORM_VISCULL_FOR_CASCADE(i, cascadeFrustumPlanes3, aabb, draw, outDrawCallsCascade3, outDrawCalls2Cascade3);
            }
        }
    }
}