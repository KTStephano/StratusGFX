STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

#include "common.glsl"
#include "aabb.glsl"

uniform vec4 frustumPlanes[6];
uniform float zfar;

layout (std430, binding = CURR_FRAME_MODEL_MATRICES_BINDING_POINT) readonly buffer inputBlock2 {
    mat4 modelTransforms[];
};

layout (std430, binding = AABB_BINDING_POINT) readonly buffer inputBlock4 {
    AABB aabbs[];
};

layout (std430, binding = VISCULL_LOD_IN_DRAW_CALLS_BINDING_POINT) readonly buffer inputBlock1 {
    DrawElementsIndirectCommand inDrawCalls[];
};

layout (std430, binding = VISCULL_LOD_OUT_DRAW_CALLS_BINDING_POINT) buffer outputBlock1 {
    DrawElementsIndirectCommand outDrawCalls[];
};

layout (std430, binding = VISCULL_LOD_SELECTED_LOD_DRAW_CALLS_BINDING_POINT) buffer outputBlock2 {
    DrawElementsIndirectCommand selectedLods[];
};

#ifdef SELECT_LOD
layout (std430, binding = VISCULL_LOD_IN_DRAW_CALLS_LOD0_BINDING_POINT) readonly buffer lod0 {
    DrawElementsIndirectCommand drawCallsLod0[];
};
layout (std430, binding = VISCULL_LOD_IN_DRAW_CALLS_LOD1_BINDING_POINT) readonly buffer lod1 {
    DrawElementsIndirectCommand drawCallsLod1[];
};
layout (std430, binding = VISCULL_LOD_IN_DRAW_CALLS_LOD2_BINDING_POINT) readonly buffer lod2 {
    DrawElementsIndirectCommand drawCallsLod2[];
};
layout (std430, binding = VISCULL_LOD_IN_DRAW_CALLS_LOD3_BINDING_POINT) readonly buffer lod3 {
    DrawElementsIndirectCommand drawCallsLod3[];
};
layout (std430, binding = VISCULL_LOD_IN_DRAW_CALLS_LOD4_BINDING_POINT) readonly buffer lod4 {
    DrawElementsIndirectCommand drawCallsLod4[];
};
layout (std430, binding = VISCULL_LOD_IN_DRAW_CALLS_LOD5_BINDING_POINT) readonly buffer lod5 {
    DrawElementsIndirectCommand drawCallsLod5[];
};
layout (std430, binding = VISCULL_LOD_IN_DRAW_CALLS_LOD6_BINDING_POINT) readonly buffer lod6 {
    DrawElementsIndirectCommand drawCallsLod6[];
};
layout (std430, binding = VISCULL_LOD_IN_DRAW_CALLS_LOD7_BINDING_POINT) readonly buffer lod7 {
    DrawElementsIndirectCommand drawCallsLod7[];
};
#endif

uniform uint numDrawCalls;
uniform vec3 viewPosition;
uniform mat4 view;
uniform mat4 prevViewProjection;

uniform sampler2D depthPyramid;
uniform int performHiZCulling;

#undef SELECT_LOD

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
        DrawElementsIndirectCommand draw = inDrawCalls[i];

    #ifdef SELECT_LOD
        DrawElementsIndirectCommand lod;
        const float firstLodDist = max(zfar, 1000.0) * 0.3;
        const float maxDist = max(zfar, 1000.0) - firstLodDist;
        const float restLodDist = maxDist / 7;

        if (dist < firstLodDist) {
            draw = drawCallsLod0[i];
        }
        // else {
        //     draw = drawCallsLod7[i];
        // }
        else if (dist < (firstLodDist + restLodDist * 1.0)) {
            draw = drawCallsLod1[i];
        }
        else if (dist < (firstLodDist + restLodDist * 2.0))  {
            draw = drawCallsLod2[i];
        }
        else if (dist < (firstLodDist + restLodDist * 3.0))  {
            draw = drawCallsLod3[i];
        }
        else if (dist < (firstLodDist + restLodDist * 4.0))  {
            draw = drawCallsLod4[i];
        }
        else if (dist < (firstLodDist + restLodDist * 5.0))  {
            draw = drawCallsLod5[i];
        }
        else if (dist < (firstLodDist + restLodDist * 6.0))  {
            draw = drawCallsLod6[i];
        }
        else {
            draw = drawCallsLod7[i];
        }

        //draw = drawCallsLod7[i];
        lod = draw;
    #endif

        int instanceCount = 1;

        if (!isAabbVisible(frustumPlanes, aabb)) {
            instanceCount = 0;
        }
        // See https://vkguide.dev/docs/gpudriven/compute_culling/
        // else if (performHiZCulling > 0) {
        //     //DrawElementsIndirectCommand prevDraw = outDrawCalls[i];
        //     //if (prevDraw.instanceCount > 0) {
        //         aabb = transformAabbAsNDCCoords(aabbs[i], prevViewProjection * modelTransforms[i]);
        //         vec3 vmin = clamp(aabb.vmin.xyz * 0.5 + vec3(0.5), vec3(0.0), vec3(1.0));
        //         vec3 vmax = clamp(aabb.vmax.xyz * 0.5 + vec3(0.5), vec3(0.0), vec3(1.0));

        //         vec2 coord1 = vec2(vmin.x, vmin.y);
        //         vec2 coord2 = vec2(vmin.x, vmax.y);
        //         vec2 coord3 = vec2(vmax.x, vmin.y);
        //         vec2 coord4 = vec2(vmax.x, vmax.y);

        //         // Calculate screenspace width/height using the uv difference between vmin and vmax
        //         float width  = (vmax.x - vmin.x) * float(textureSize(depthPyramid, 0).x);
        //         float height = (vmax.y - vmin.y) * float(textureSize(depthPyramid, 0).y);

        //         // Compute mip level where AABB is close to the size of 1 pixel
        //         float level = floor(log2(max(width, height)));

        //         float depth1 = textureLod(depthPyramid, coord1, level).x;
        //         float depth2 = textureLod(depthPyramid, coord2, level).x;
        //         float depth3 = textureLod(depthPyramid, coord3, level).x;
        //         float depth4 = textureLod(depthPyramid, coord4, level).x;

        //         float depth = max(max(depth1, depth2), max(depth3, depth4));

        //         // vec4 minDepth = viewProjection * vec4(0, 0, dist, 1);
        //         // minDepth.xyz /= minDepth.w;
        //         // minDepth.xyz = minDepth.xyz * 0.5 + vec3(0.5);

        //         bool invalid = (vmin.z <= depth == false) && (vmin.z > depth == false);

        //         instanceCount = 1;//int(vmin.z <= depth || vmin.z >= depth);
        //     //}
        // }

        draw.instanceCount = instanceCount;

        outDrawCalls[i] = draw;

    #ifdef SELECT_LOD
        selectedLods[i] = lod;
    #endif
    }
}