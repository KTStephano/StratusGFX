STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

// This is to determine which lights are visible to which parts of the screen
// where the screen has been partitioned into a 2D grid of tiles

// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
// Also see https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction
layout (local_size_x = 4, local_size_y = 4, local_size_z = 1) in;

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
shared vec3 averageLocalPosition;
shared vec3 averageLocalNormal;
shared float lightDistanceRadiusRatios[MAX_TOTAL_VPLS_PER_FRAME];

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
    int baseTileIndex = tileIndex * MAX_VPLS_PER_TILE;
    vec3 fragPos = texture(gPosition, texCoords).xyz;
    vec3 normal = normalize(texture(gNormal, texCoords).rgb * 2.0 - vec3(1.0)); // [0, 1] -> [-1, 1]

    // Each thread will now add their local contribution to position and normal to compute averages
    localPositions[gl_LocalInvocationIndex] = fragPos;
    localNormals[gl_LocalInvocationIndex]   = normal;

    if (gl_LocalInvocationIndex == 0) {
        vplNumVisiblePerTile[tileIndex] = 0;
    }

    // Wait for all local threads to get here
    barrier();

    // Compute average on first local work group thread
    if (gl_LocalInvocationIndex == 0) {
        averageLocalPosition = vec3(0.0);
        averageLocalNormal   = vec3(0.0);

        for (int i = 0; i < localWorkGroupSize; ++i) {
            averageLocalPosition += localPositions[i];
            averageLocalNormal   += localNormals[i];
        }

        averageLocalPosition /= float(localWorkGroupSize);
        averageLocalNormal   /= float(localWorkGroupSize);
        averageLocalNormal    = normalize(averageLocalNormal);
    }

    // Wait for all local threads to get here
    barrier();

    // Compute the light distance to radius ratios in parallel
    for (uint i = gl_LocalInvocationIndex; i < numVisible; i += localWorkGroupSize) {
        int lightIndex = vplVisibleIndex[i];
        vec3 lightPosition = lightPositions[lightIndex].xyz;
        float distance = length(lightPosition - averageLocalPosition);
        float radius = lightRadii[lightIndex];
        float ratio = distance / radius;

        lightDistanceRadiusRatios[lightIndex] = ratio;
    }

    // Wait for all local threads to get here
    barrier();

    // Determine light visibility for this tile
    if (gl_LocalInvocationIndex == 0) {
        int numVisibleThisTile = 0;
        int indicesVisibleThisTile[MAX_VPLS_PER_TILE];
        float distancesVisibleThisTile[MAX_VPLS_PER_TILE];

        for (int i = 0; i < MAX_VPLS_PER_TILE; ++i) {
            indicesVisibleThisTile[i] = i;
            distancesVisibleThisTile[i] = FLOAT_MAX;
        }

        for (int i = 0; i < numVisible; ++i) {
            //float intensity = length()
            int lightIndex = vplVisibleIndex[i];
            vec3 lightPosition = lightPositions[lightIndex].xyz;
            // Make sure the light is in the direction of the plane+normal. If n*(a-p) < 0, the point is on the other side of the plane.
            // If 0 the point is on the plane. If > 0 then the point is on the side of the plane visible along the normal's direction.
            // See https://math.stackexchange.com/questions/1330210/how-to-check-if-a-point-is-in-the-direction-of-the-normal-of-a-plane
            vec3 lightMinusFrag = lightPosition - averageLocalPosition;
            float sideCheck = dot(averageLocalNormal, lightMinusFrag);
            if (sideCheck < 0) continue;

            float radius = lightRadii[lightIndex];
            float ratio = lightDistanceRadiusRatios[lightIndex];

            if (ratio > 1.0) continue;

            float distance = ratio;
            for (int ii = 0; ii < MAX_VPLS_PER_TILE; ++ii) {
                if (distance < distancesVisibleThisTile[ii]) {
                    //if (ratio < 0.1) {
                        float shadowFactor = calculateShadowValue1Sample(shadowCubeMaps[lightIndex], radius, averageLocalPosition, lightPosition, dot(lightMinusFrag, averageLocalNormal));
                        // // Light can't see current surface
                        if (shadowFactor > 0.0) break;
                    //}

                    SHUFFLE_DOWN(indicesVisibleThisTile, ii)
                    SHUFFLE_DOWN(distancesVisibleThisTile, ii)
                    indicesVisibleThisTile[ii] = lightIndex;
                    distancesVisibleThisTile[ii] = distance;
                    if (numVisibleThisTile < MAX_VPLS_PER_TILE) {
                        ++numVisibleThisTile;
                    }
                    break;
                }
            }
        }
        
        vplNumVisiblePerTile[tileIndex] = numVisibleThisTile;
        for (int i = 0; i < numVisibleThisTile; ++i) {
            vplIndicesVisiblePerTile[baseTileIndex + i] = indicesVisibleThisTile[i];
        }
    }

    // vplNumVisiblePerTile[tileIndex] = numVisibleThisTile;
    // for (int i = 0; i < numVisibleThisTile; ++i) {
    //     vplIndicesVisiblePerTile[baseTileIndex + i] = indicesVisibleThisTile[i];
    // }
}