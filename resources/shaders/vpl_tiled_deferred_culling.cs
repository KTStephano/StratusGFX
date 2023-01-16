STRATUS_GLSL_VERSION

// This is to determine which lights are visible to which parts of the screen
// where the screen has been partitioned into a 2D grid of tiles

// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
// Also see https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction
//
// 16:9 aspect ratio
layout (local_size_x = 16, local_size_y = 9, local_size_z = 1) in;

#include "vpl_tiled_deferred_culling.glsl"

// GBuffer information
uniform sampler2D gPosition;

// for vec2 with std140 it always begins on a 2*4 = 8 byte boundary
// for vec3, vec4 with std140 it always begins on a 4*4=16 byte boundary

// See https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Memory_layout
// for information on layout std140 and std430. In std140 floats aren't guaranteed
// to be packed and so float arrays in OpenGL are not the same as float arrays in C/C++.
//
// This changes with std430 where it enforces equivalency between OpenGL and C/C++ float arrays
// by tightly packing them.
layout (std430, binding = 0) readonly buffer vplLightData {
    VirtualPointLight lightData[];
};

layout (std430, binding = 1) readonly buffer numVisibleVPLs {
    int numVisible;
};

layout (std430, binding = 3) readonly buffer visibleVplTable {
    int vplVisibleIndex[];
};

uniform int viewportWidth;
uniform int viewportHeight;

layout (std430, binding = 5) writeonly buffer outputBlock1 {
    int vplIndicesVisiblePerTile[];
};

layout (std430, binding = 6) writeonly buffer outputBlock2 {
    int vplNumVisiblePerTile[];
};

shared int activeLightBitmask[MAX_TOTAL_VPLS_PER_FRAME];

void main() {
    // If we're the first in the work group initialize the bitmask
    if (gl_LocalInvocationIndex == 0) {
        for (int i = 0; i < MAX_TOTAL_VPLS_PER_FRAME; ++i) {
           activeLightBitmask[i] = 0;
        }
    }

    // Wait for the rest
    barrier();

    // For example, 1920x1080 would result in 120x120 tiles
    uvec2 numTiles = gl_NumWorkGroups.xy;
    uvec2 tileCoords = gl_WorkGroupID.xy;
    uvec2 pixelCoords = tileCoords * gl_WorkGroupSize.xy + gl_LocalInvocationID.xy;
    vec2 texCoords = vec2(pixelCoords) / vec2(viewportWidth, viewportHeight);
    int baseTileIndex = int(tileCoords.x * MAX_VPLS_PER_TILE + tileCoords.y * numTiles.x * MAX_VPLS_PER_TILE);
    vec3 fragPos = texture(gPosition, texCoords).rgb;

    for (int i = 0; i < numVisible; ++i) {
        int lightIndex = vplVisibleIndex[i];
        VirtualPointLight vpl = lightData[lightIndex];
        float distance = length(vpl.lightPosition.xyz - fragPos);
        float radius = vpl.lightRadius;
        if (distance > radius) continue;
        activeLightBitmask[lightIndex] = 1;
    }

    // Wait for work group to finish
    barrier();

    // If we're the first in the work group update num visible
    if (gl_LocalInvocationIndex == 0) {
        int numVisible = 0;
        int next = 0;
        for (int i = 0; i < MAX_TOTAL_VPLS_PER_FRAME; ++i) {
            int bitmaskVal = activeLightBitmask[i];
            numVisible += bitmaskVal;
            if (bitmaskVal > 0) {
                vplIndicesVisiblePerTile[baseTileIndex + next] = i;
                next += 1;
            }
            if (numVisible >= MAX_VPLS_PER_TILE) break;
        }
        vplNumVisiblePerTile[tileCoords.x + tileCoords.y * numTiles.x] = numVisible;
    }
}

/*
void main() {
    uvec2 numTiles = uvec2(viewportWidth, viewportHeight);
    uvec2 tileCoords = gl_GlobalInvocationID.xy;
    uvec2 pixelCoords = tileCoords;
    // See https://stackoverflow.com/questions/40574677/how-to-normalize-image-coordinates-for-texture-space-in-opengl
    vec2 texCoords = (vec2(pixelCoords) + vec2(0.5)) / vec2(viewportWidth, viewportHeight);
    int tileIndex = int(tileCoords.x + tileCoords.y * numTiles.x);
    int baseTileIndex = tileIndex * MAX_VPLS_PER_TILE;
    vec3 fragPos = texture(gPosition, texCoords).xyz;

    int activeLightsThisTile = 0;

    const float distOffset = 0.25;
    float minDist = 0.0;
    float maxDist = distOffset;

    // If we don't have many VPLs, just check against the entire distance range right away
    if (numVisible < MAX_VPLS_PER_TILE) {
        maxDist = 1.0;
    }

    while (maxDist < 1.05 && activeLightsThisTile < MAX_VPLS_PER_TILE && activeLightsThisTile < numVisible) {
        for (int i = 0; i < numVisible && activeLightsThisTile < MAX_VPLS_PER_TILE; ++i) {
            int lightIndex = vplVisibleIndex[i];
            VirtualPointLight vpl = lightData[lightIndex];
            vec3 lightPosition = vpl.lightPosition.xyz;
            float lightRadius = vpl.lightRadius;
            float distance = length(lightPosition - fragPos) / lightRadius;
            if (distance >= minDist && distance < maxDist) {
                //vplNumVisiblePerTile[tileIndex] = int(distance * 100);
                vplIndicesVisiblePerTile[baseTileIndex + activeLightsThisTile] = lightIndex;
                activeLightsThisTile += 1;
            }
        }
        minDist += distOffset;
        maxDist += distOffset;
    }

    vplNumVisiblePerTile[tileIndex] = activeLightsThisTile;
    barrier();
}
*/

/*
void main() {
    uvec2 numTiles = uvec2(viewportWidth, viewportHeight);
    uvec2 tileCoords = gl_GlobalInvocationID.xy;
    uvec2 pixelCoords = tileCoords;
    // See https://stackoverflow.com/questions/40574677/how-to-normalize-image-coordinates-for-texture-space-in-opengl
    vec2 texCoords = (vec2(pixelCoords) + vec2(0.5)) / vec2(viewportWidth, viewportHeight);
    int tileIndex = int(tileCoords.x + tileCoords.y * numTiles.x);
    int baseTileIndex = tileIndex * MAX_VPLS_PER_TILE;
    vec3 fragPos = texture(gPosition, texCoords).xyz;

    int activeLightsThisTile = 0;
    for (int i = 0; i < numVisible && activeLightsThisTile < MAX_VPLS_PER_TILE; ++i) {
        int lightIndex = vplVisibleIndex[i];
        VirtualPointLight vpl = lightData[lightIndex];
        vec3 lightPosition = vpl.lightPosition.xyz;
        float distance = length(lightPosition - fragPos);
        if (distance < vpl.lightRadius) {
            //vplNumVisiblePerTile[tileIndex] = int(distance * 100);
            vplIndicesVisiblePerTile[baseTileIndex + activeLightsThisTile] = lightIndex;
            activeLightsThisTile += 1;
        }
    }

    vplNumVisiblePerTile[tileIndex] = activeLightsThisTile;
    barrier();
}
*/

/*
void main() {
    uvec2 numTiles = uvec2(viewportWidth, viewportHeight);
    uvec2 tileCoords = gl_GlobalInvocationID.xy;
    uvec2 pixelCoords = tileCoords;
    // See https://stackoverflow.com/questions/40574677/how-to-normalize-image-coordinates-for-texture-space-in-opengl
    vec2 texCoords = (vec2(pixelCoords) + vec2(0.5)) / vec2(viewportWidth, viewportHeight);
    int tileIndex = int(tileCoords.x + tileCoords.y * numTiles.x);
    int baseTileIndex = tileIndex * MAX_VPLS_PER_TILE;
    vec3 fragPos = texture(gPosition, texCoords).xyz;

    int activeLightsThisTile = 0;

    for (int i = 0; i < numVisible && activeLightsThisTile < MAX_VPLS_PER_TILE; ++i) {
        int lightIndex = vplVisibleIndex[i];
        VirtualPointLight vpl = lightData[lightIndex];
        vec3 lightPosition = vpl.lightPosition.xyz;
        float lightRadius = vpl.lightRadius / 2.0;
        float distance = length(lightPosition - fragPos) / lightRadius;
        if (distance >= 0.0 && distance < 1.0) {
            //vplNumVisiblePerTile[tileIndex] = int(distance * 100);
            vplIndicesVisiblePerTile[baseTileIndex + activeLightsThisTile] = lightIndex;
            activeLightsThisTile += 1;
        }
    }

    vplNumVisiblePerTile[tileIndex] = activeLightsThisTile;
    barrier();
}
*/