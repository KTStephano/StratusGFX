STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

// This is to determine which lights are visible to which parts of the screen
// where the screen has been partitioned into a 2D grid of tiles

// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
// Also see https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction

// Each thread group processes 2 * 4 = 8 pixels
layout (local_size_x = 2, local_size_y = 6, local_size_z = 1) in;
//layout(local_size_x = 5, local_size_y = 5, local_size_z = 1) in;

#include "vpl_common.glsl"
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
layout (std430, binding = 0) readonly buffer inputBlock1 {
    VplData lightData[];
};

layout (std430, binding = 1) writeonly buffer outputBlock1 {
    VplStage1PerTileOutputs vplNumVisiblePerTile[];
};

layout (std430, binding = 11) readonly buffer inputBlock2 {
    samplerCube shadowCubeMaps[];
};                           

shared vec3 localPositions[gl_WorkGroupSize.x * gl_WorkGroupSize.y];
shared vec3 localNormals[gl_WorkGroupSize.x * gl_WorkGroupSize.y];

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
    vec3 fragPos = textureLod(gPosition, texCoords, 0).xyz;
    vec3 normal = normalize(textureLod(gNormal, texCoords, 0).rgb * 2.0 - vec3(1.0)); // [0, 1] -> [-1, 1]

    // Each thread will now add their local contribution to position and normal to compute averages
    localPositions[gl_LocalInvocationIndex] = fragPos;
    localNormals[gl_LocalInvocationIndex]   = normal;

    // Wait for all local threads to get here
    barrier();

    // Compute average on first local work group thread
    if (gl_LocalInvocationIndex == 0) {
        vec3 averageLocalPosition = vec3(0.0);
        vec3 averageLocalNormal   = vec3(0.0);

        for (int i = 0; i < localWorkGroupSize; ++i) {
            averageLocalPosition += localPositions[i];
            averageLocalNormal   += localNormals[i];
        }

        averageLocalPosition /= float(localWorkGroupSize);
        averageLocalNormal   /= float(localWorkGroupSize);
        averageLocalNormal    = normalize(averageLocalNormal);

        vplNumVisiblePerTile[tileIndex].averageLocalPosition = vec4(averageLocalPosition, 1.0);
        vplNumVisiblePerTile[tileIndex].averageLocalNormal   = vec4(averageLocalNormal, 1.0);
    }
}