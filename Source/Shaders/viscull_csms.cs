STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

#include "common.glsl"
#include "aabb.glsl"

uniform uint numDrawCalls;
uniform mat4 cascadeViewProj[4];

layout (std430, binding = 2) readonly buffer inputBlock3 {
    mat4 modelTransforms[];
};

layout (std430, binding = 3) readonly buffer inputBlock4 {
    AABB aabbs[];
};

// Cascades 0 and 1
layout (std430, binding = 1) readonly buffer inputBlock1 {
    DrawElementsIndirectCommand cascade0DrawCalls[];
};

// Cascades 2 and 3
layout (std430, binding = 4) readonly buffer inputBlock2 {
    DrawElementsIndirectCommand cascade123DrawCalls[];
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

shared vec4 cascadeFrustumPlanes0[6];
shared vec4 cascadeFrustumPlanes1[6];
shared vec4 cascadeFrustumPlanes2[6];
shared vec4 cascadeFrustumPlanes3[6];

void main() {
    // Defines local work group from layout local size tag above
    uint localWorkGroupSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y;

    // Extract world-space frustum planes
#define INITIALIZE_CASCADE_PLANES(index, planes)            \
    if (gl_LocalInvocationIndex == index) {                 \
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

// There is a bug when using aabb culling with cascades - not sure what causes it,
// but there are false negatives (culled but should have been kept)
#define PERFORM_VISCULL_FOR_CASCADE(index, planes, aabb, draw, out) \
    if (!isAabbVisible(planes, aabb)) {                             \
        draw.instanceCount = 0;                                     \
    } else {                                                        \
        draw.instanceCount = 1;                                     \
    }                                                               \
    out[index] = draw;

#define UNCONDITIONAL_SET_VISIBLE(index, draw, out)                 \
    draw.instanceCount = 1;                                         \
    out[index] = draw;                                               
   
    for (uint i = gl_LocalInvocationIndex; i < numDrawCalls; i += localWorkGroupSize) {
        AABB aabb = transformAabb(aabbs[i], modelTransforms[i]);

        // Cascade 0
        DrawElementsIndirectCommand draw = cascade0DrawCalls[i];
        //PERFORM_VISCULL_FOR_CASCADE(i, cascadeFrustumPlanes0, aabb, draw, outDrawCallsCascade0);
        UNCONDITIONAL_SET_VISIBLE(i, draw, outDrawCallsCascade0);

        // Cascades 1, 2, 3
        draw = cascade123DrawCalls[i];
        UNCONDITIONAL_SET_VISIBLE(i, draw, outDrawCallsCascade1);
        UNCONDITIONAL_SET_VISIBLE(i, draw, outDrawCallsCascade2);
        UNCONDITIONAL_SET_VISIBLE(i, draw, outDrawCallsCascade3);
        //PERFORM_VISCULL_FOR_CASCADE(i, cascadeFrustumPlanes1, aabb, draw, outDrawCallsCascade1);
        //PERFORM_VISCULL_FOR_CASCADE(i, cascadeFrustumPlanes2, aabb, draw, outDrawCallsCascade2);
        //PERFORM_VISCULL_FOR_CASCADE(i, cascadeFrustumPlanes3, aabb, draw, outDrawCallsCascade3);
    }
}