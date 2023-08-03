STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

#include "common.glsl"
#include "aabb.glsl"

uniform uint numDrawCalls;

uniform mat4 viewProj[6];

layout (std430, binding = 2) readonly buffer inputBlock4 {
    mat4 modelTransforms[];
};

layout (std430, binding = 3) readonly buffer inputBlock7 {
    AABB aabbs[];
};

// Draw calls for face culling modes 1-3
layout (std430, binding = 1) readonly buffer inputBlock1 {
    DrawElementsIndirectCommand inDrawCalls[];
};

// Outputs for first face culling
layout (std430, binding = 4) buffer outputBlock1 {
    DrawElementsIndirectCommand outDrawCalls0[];
};

layout (std430, binding = 5) buffer outputBlock2 {
    DrawElementsIndirectCommand outDrawCalls1[];
};

layout (std430, binding = 6) buffer outputBlock3 {
    DrawElementsIndirectCommand outDrawCalls2[];
};

layout (std430, binding = 7) buffer outputBlock4 {
    DrawElementsIndirectCommand outDrawCalls3[];
};

layout (std430, binding = 8) buffer outputBlock5 {
    DrawElementsIndirectCommand outDrawCalls4[];
};

layout (std430, binding = 9) buffer outputBlock6 {
    DrawElementsIndirectCommand outDrawCalls5[];
};

shared vec4 frustumPlanes0[6];
shared vec4 frustumPlanes1[6];
shared vec4 frustumPlanes2[6];
shared vec4 frustumPlanes3[6];
shared vec4 frustumPlanes4[6];
shared vec4 frustumPlanes5[6];

void main() {
    // Defines local work group from layout local size tag above
    uint localWorkGroupSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y;

    // Extract world-space frustum planes
#define INITIALIZE_CASCADE_PLANES(index, planes)            \
    if (gl_LocalInvocationIndex == index) {                 \
        mat4 vpt = transpose(viewProj[index]);              \
        planes[0] = vpt[3] + vpt[0];                        \
        planes[1] = vpt[3] - vpt[0];                        \
        planes[2] = vpt[3] + vpt[1];                        \
        planes[3] = vpt[3] - vpt[1];                        \
        planes[4] = vpt[3] + vpt[2];                        \
        planes[5] = vpt[3] - vpt[2];                        \
    }

    INITIALIZE_CASCADE_PLANES(0, frustumPlanes0);
    INITIALIZE_CASCADE_PLANES(1, frustumPlanes1);
    INITIALIZE_CASCADE_PLANES(2, frustumPlanes2);
    INITIALIZE_CASCADE_PLANES(3, frustumPlanes3);
    INITIALIZE_CASCADE_PLANES(4, frustumPlanes4);
    INITIALIZE_CASCADE_PLANES(5, frustumPlanes5);

    barrier();

#define PERFORM_VISCULL_FOR_DIRECTION(index, planes, aabb, draw, out) \
    if (!isAabbVisible(planes, aabb)) {                               \
        draw.instanceCount = 0;                                       \
    } else {                                                          \
        draw.instanceCount = 1;                                       \
    }                                                                 \
    out[index] = draw;
   
    // First face
    for (uint i = gl_LocalInvocationIndex; i < numDrawCalls; i += localWorkGroupSize) {
        AABB aabb = transformAabb(aabbs[i], modelTransforms[i]);
        DrawElementsIndirectCommand draw = inDrawCalls[i];

        PERFORM_VISCULL_FOR_DIRECTION(i, frustumPlanes0, aabb, draw, outDrawCalls0);
        PERFORM_VISCULL_FOR_DIRECTION(i, frustumPlanes1, aabb, draw, outDrawCalls1);
        PERFORM_VISCULL_FOR_DIRECTION(i, frustumPlanes2, aabb, draw, outDrawCalls2);
        PERFORM_VISCULL_FOR_DIRECTION(i, frustumPlanes3, aabb, draw, outDrawCalls3);
        PERFORM_VISCULL_FOR_DIRECTION(i, frustumPlanes4, aabb, draw, outDrawCalls4);
        PERFORM_VISCULL_FOR_DIRECTION(i, frustumPlanes5, aabb, draw, outDrawCalls5);
    }
}