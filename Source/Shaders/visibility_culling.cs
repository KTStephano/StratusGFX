STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

#include "common.glsl"
#include "aabb.glsl"

layout (std430, binding = 2) readonly buffer inputBlock2 {
    mat4 modelTransforms[];
};

layout (std430, binding = 3) readonly buffer inputBlock4 {
    AABB aabbs[];
};

layout (std430, binding = 1) buffer outputBlock1 {
    DrawElementsIndirectCommand drawCalls[];
};

layout (std430, binding = 13) buffer outputBlock2 {
    DrawElementsIndirectCommand selectedLods[];
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
    layout (std430, binding = 9) readonly buffer lod4 {
        DrawElementsIndirectCommand drawCallsLod4[];
    };
    layout (std430, binding = 10) readonly buffer lod5 {
        DrawElementsIndirectCommand drawCallsLod5[];
    };
    layout (std430, binding = 11) readonly buffer lod6 {
        DrawElementsIndirectCommand drawCallsLod6[];
    };
    layout (std430, binding = 12) readonly buffer lod7 {
        DrawElementsIndirectCommand drawCallsLod7[];
    };
#endif

uniform uint numDrawCalls;
uniform vec3 viewPosition;
uniform mat4 view;

// See https://stackoverflow.com/questions/5254838/calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
float distanceFromPointToAABB(in AABB aabb, vec3 point) {
    float dx = max(aabb.vmin.x - point.x, max(0.0, point.x - aabb.vmax.x));
    float dy = max(aabb.vmin.y - point.y, max(0.0, point.y - aabb.vmax.y));
    float dz = max(aabb.vmin.z - point.z, max(0.0, point.z - aabb.vmax.z));

    return sqrt(dx * dx + dy * dy + dz + dz);
}

void main() {
    // Defines local work group from layout local size tag above
    uint localWorkGroupSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
   
    for (uint i = gl_LocalInvocationIndex; i < numDrawCalls; i += localWorkGroupSize) {
        AABB aabb = transformAabb(aabbs[i], modelTransforms[i]);
        // World space center
        //vec3 center = (aabb.vmin.xyz + aabb.vmax.xyz) * 0.5;
        // Defines a ray originating from the camera moving towards the center of the AABB
        float dist = distanceFromPointToAABB(aabb, viewPosition);
        //center = center - viewPosition; //(view * vec4(center, 1.0)).xyz;
        //float dist = length((view * vec4(center, 1.0)).xyz);//abs(center.z);
        DrawElementsIndirectCommand draw = drawCalls[i];

    #ifdef SELECT_LOD
        DrawElementsIndirectCommand lod;
        float initialDist = 50.0;
        if (dist < initialDist) {
            draw = drawCallsLod0[i];
        }
        // else {
        //     draw = drawCallsLod7[i];
        // }
        else if (dist < initialDist * 2.0) {
            draw = drawCallsLod1[i];
        }
        else if (dist < initialDist * 3.0) {
            draw = drawCallsLod2[i];
        }
        else if (dist < initialDist * 4.0) {
            draw = drawCallsLod3[i];
        }
        else if (dist < initialDist * 5.0) {
            draw = drawCallsLod4[i];
        }
        else if (dist < initialDist * 6.0) {
            draw = drawCallsLod5[i];
        }
        else if (dist < initialDist * 7.0) {
            draw = drawCallsLod6[i];
        }
        else {
            draw = drawCallsLod7[i];
        }

        lod = draw;
    #endif

        if (!isAabbVisible(aabb)) {
            draw.instanceCount = 0;
        }
        else {
            draw.instanceCount = 1;
        }

        drawCalls[i] = draw;

    #ifdef SELECT_LOD
        selectedLods[i] = lod;
    #endif
    }
}