STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

#include "common.glsl"
#include "aabb.glsl"

layout (std430, binding = 2) readonly buffer inputBlock2 {
    mat4 modelTransforms[];
};

layout (std430, binding = 4) readonly buffer inputBlock3 {
    mat4 globalTransforms[];
};

layout (std430, binding = 3) readonly buffer inputBlock4 {
    AABB aabbs[];
};

layout (std430, binding = 1) buffer outputBlock1 {
    DrawElementsIndirectCommand drawCalls[];
};

#ifdef SELECT_LOD
layout (std430, binding = 5) readonly buffer lod0 {
    DrawElementsIndirectCommand drawCallsLod0[];
};
layout (std430, binding = 6) readonly buffer lod1 {
    DrawElementsIndirectCommand drawCallsLod1[];
};
layout (std430, binding = 7) readonly buffer lod2 {
    DrawElementsIndirectCommand drawCallsLod2[];
};
layout (std430, binding = 8) readonly buffer lod3 {
    DrawElementsIndirectCommand drawCallsLod3[];
};
#endif

uniform uint numDrawCalls;
uniform vec3 viewPosition;
uniform mat4 view;

void main() {
    // Defines local work group from layout local size tag above
    uint localWorkGroupSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
   
    for (uint i = gl_LocalInvocationIndex; i < numDrawCalls; i += localWorkGroupSize) {
        AABB aabb = transformAabb(aabbs[i], globalTransforms[i]);
        vec3 center = (aabb.vmin.xyz + aabb.vmax.xyz) * 0.5;
        center = (view * vec4(center, 1.0)).xyz;
        float dist = length(center);
        DrawElementsIndirectCommand draw = drawCalls[i];
        if (!isAabbVisible(aabb)) {
            draw.instanceCount = 0;
        }
        else {
            #ifdef SELECT_LOD
            if (dist < 200) {
                draw = drawCallsLod0[i];
            }
            else if (dist < 300) {
                draw = drawCallsLod1[i];
            }
            else if (dist < 400) {
                draw = drawCallsLod2[i];
            }
            else {
                draw = drawCallsLod3[i];
            }
            #endif
            draw.instanceCount = 1;
        }

        drawCalls[i] = draw;
    }
}