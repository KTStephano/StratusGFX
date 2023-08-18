STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#include "common.glsl"
#include "aabb.glsl"

uniform mat4 cascadeProjectionView;

uniform uint frameCount;

uniform uint numDrawCalls;

layout (r32ui) readonly uniform uimage2D currFramePageResidencyTable;

layout (std430, binding = 2) readonly buffer inputBlock2 {
    mat4 modelTransforms[];
};

layout (std430, binding = 3) readonly buffer inputBlock4 {
    AABB aabbs[];
};

layout (std430, binding = 1) readonly buffer inputBlock1 {
    DrawElementsIndirectCommand inDrawCalls[];
};

layout (std430, binding = 4) buffer outputBlock1 {
    DrawElementsIndirectCommand outDrawCalls[];
};

layout (std430, binding = 5) buffer outputBlock2 {
    int outputDrawCalls;
};

shared ivec2 residencyTableSize;
shared ivec2 maxResidencyTableIndex;
shared bool validPage;

void main() {
    ivec2 baseTileCoords = ivec2(gl_WorkGroupID.xy);

    if (gl_LocalInvocationID == 0) {
        residencyTableSize = imageSize(currFramePageResidencyTable).xy;
        maxResidencyTableIndex = residencyTableSize - ivec2(1.0);

        uint pageStatus = uint(imageLoad(currFramePageResidencyTable, baseTileCoords).r);
        validPage = (pageStatus == frameCount);
    }

    barrier();

    // Page is not needed for this frame - skip entire work group
    if (validPage == false) {
        return;
    }

    // Defines local work group from layout local size tag above
    uint localWorkGroupSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y;

    vec4 corners[8];
    vec2 texCoords;
    ivec2 currentTileCoords;

    for (uint i = gl_LocalInvocationIndex; i < numDrawCalls; i += localWorkGroupSize) {
        DrawElementsIndirectCommand draw = inDrawCalls[i];
        if (draw.instanceCount == 0) {
            continue;
        }

        AABB aabb = transformAabb(aabbs[i], cascadeProjectionView * modelTransforms[i]);
        computeCorners(aabb, corners);

        // TODO: Need to check if this tile is inside the bounding box!
        for (int j = 0; j < 8; ++j) {
            // Maps from [-1, 1] to [0, 1]
            texCoords = corners[j].xy * 0.5 + 0.5;
            currentTileCoords = ivec2(texCoords * vec2(maxResidencyTableIndex));

            // Check if the corner lies within this tile
            if (currentTileCoords == baseTileCoords) {
                uint prev = atomicExchange(outDrawCalls[i].instanceCount, 1);

                // if (prev == 0) {
                //     atomicAdd(outputDrawCalls, 1);
                // }

                break;
            }
        }
    }
}