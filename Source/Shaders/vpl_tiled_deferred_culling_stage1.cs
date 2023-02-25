STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

// Stage 1 of this two-stage compute pipeline determines which lights are visible to which
// tile where tiles are each defined as having 144 pixels (16*9).

// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
// Also see https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction
//
// 16:9 aspect ratio
layout (local_size_x = 16, local_size_y = 9, local_size_z = 1) in;

#include "vpl_tiled_deferred_culling.glsl"
#include "common.glsl"
#include "pbr.glsl"

// GBuffer information
uniform sampler2D gPosition;
uniform sampler2D gNormal;

// for vec2 with std140 it always begins on a 2*4 = 8 byte boundary
// for vec3, vec4 with std140 it always begins on a 4*4=16 byte boundary

// See https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Memory_layout
// for information on layout std140 and std430. In std140 floats aren't guaranteed
// to be packed and so float arrays in OpenGL are not the same as float arrays in C/C++.
//
// This changes with std430 where it enforces equivalency between OpenGL and C/C++ float arrays
// by tightly packing them.
layout (std430, binding = 0) readonly buffer vplLightPositions {
    vec4 lightPositions[];
};

layout (std430, binding = 7) readonly buffer vplLightRadii {
    float lightRadii[];
};

layout (std430, binding = 1) readonly buffer numVisibleVPLs {
    int numVisible;
};

layout (std430, binding = 3) readonly buffer visibleVplTable {
    int vplVisibleIndex[];
};

layout (std430, binding = 10) readonly buffer vplColors {
    vec4 lightColors[];
};

layout (std430, binding = 5) writeonly buffer outputBlock1 {
    int vplIndicesVisiblePerTile[];
};

layout (std430, binding = 6) writeonly buffer outputBlock2 {
    int vplNumVisiblePerTile[];
};

layout (std430, binding = 11) readonly buffer vplShadows {
    samplerCube shadowCubeMaps[];
};                          

shared vec3 localPositions[gl_WorkGroupSize.x * gl_WorkGroupSize.y];
shared vec3 localNormals[gl_WorkGroupSize.x * gl_WorkGroupSize.y];
shared vec3 averageLocalPosition;
shared vec3 averageLocalNormal;
shared bool lightVisible[MAX_TOTAL_VPLS_PER_FRAME];
shared float lightVisibleDistances[MAX_TOTAL_VPLS_PER_FRAME];

void main() {
    // Defines local work group from layout local size tag above
    uint localWorkGroupSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    uvec2 numTiles = gl_NumWorkGroups.xy;
    uvec2 tileCoords = gl_WorkGroupID.xy;
    uvec2 viewportWidthHeight = gl_NumWorkGroups.xy * gl_WorkGroupSize.xy;
    uvec2 pixelCoords = gl_GlobalInvocationID.xy;
    // See https://stackoverflow.com/questions/40574677/how-to-normalize-image-coordinates-for-texture-space-in-opengl
    vec2 texCoords = (vec2(pixelCoords) + vec2(0.5)) / vec2(viewportWidthHeight.x, viewportWidthHeight.y);

    int tileIndex = int(tileCoords.x + tileCoords.y * numTiles.x);
    int baseTileIndex = tileIndex * MAX_TOTAL_VPLS_PER_FRAME;
    vec3 fragPos = texture(gPosition, texCoords).xyz;
    vec3 normal = normalize(texture(gNormal, texCoords).rgb * 2.0 - vec3(1.0)); // [0, 1] -> [-1, 1]

    // Each thread will now add their local contribution to position and normal to compute averages
    localPositions[gl_LocalInvocationIndex] = fragPos;
    localNormals[gl_LocalInvocationIndex]   = normal;

    if (gl_LocalInvocationIndex == 0) {
        vplNumVisiblePerTile[tileIndex] = 0;
        for (int i = 0; i < MAX_TOTAL_VPLS_PER_FRAME; ++i) {
            lightVisible[i] = false;
        }
    }

    // Wait for all local threads to get here
    barrier();

    // The next few steps are an incremental sum using multiple threads
    if (mod(gl_LocalInvocationIndex, 2) == 0) {
        localPositions[gl_LocalInvocationIndex] += localPositions[gl_LocalInvocationIndex + 1];
        localNormals[gl_LocalInvocationIndex]   += localNormals[gl_LocalInvocationIndex   + 1];
    }

    barrier();

    if (mod(gl_LocalInvocationIndex, 4) == 0) {
        localPositions[gl_LocalInvocationIndex] += localPositions[gl_LocalInvocationIndex + 2];
        localNormals[gl_LocalInvocationIndex]   += localNormals[gl_LocalInvocationIndex   + 2];
    }

    barrier();

    if (mod(gl_LocalInvocationIndex, 8) == 0) {
        localPositions[gl_LocalInvocationIndex] += localPositions[gl_LocalInvocationIndex + 4];
        localNormals[gl_LocalInvocationIndex]   += localNormals[gl_LocalInvocationIndex   + 4];
    }

    barrier();

    // Sum the rest
    if (gl_LocalInvocationIndex == 0) {
        averageLocalPosition = vec3(0.0);
        averageLocalNormal   = vec3(0.0);
        for (int i = 0; i < localWorkGroupSize; i += 8) {
            averageLocalPosition += localPositions[i];
            averageLocalNormal   += localNormals[i];
        }
        averageLocalPosition /= float(localWorkGroupSize);
        averageLocalNormal   /= float(localWorkGroupSize);
        averageLocalNormal    = normalize(averageLocalNormal);
    }

    // Wait for all local threads to get here
    barrier();

    // Determine which lights are visible
    for (uint i = gl_LocalInvocationIndex; i < numVisible; i += localWorkGroupSize) {
        int lightIndex = vplVisibleIndex[i];
        vec3 lightPosition = lightPositions[lightIndex].xyz;
        float distance = length(lightPosition - fragPos);
        float radius = lightRadii[lightIndex];
        float ratio = distance / radius;

        if (ratio > 1.0) continue;

        lightVisible[lightIndex] = true;
        lightVisibleDistances[lightIndex] = distance;
    }

    barrier();
}