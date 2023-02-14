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
#include "common.glsl"

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

uniform int viewportWidth;
uniform int viewportHeight;

layout (std430, binding = 5) writeonly buffer outputBlock1 {
    int vplIndicesVisiblePerTile[];
};

layout (std430, binding = 6) writeonly buffer outputBlock2 {
    int vplNumVisiblePerTile[];
};

/*
shared int activeLightMarker[MAX_TOTAL_VPLS_PER_FRAME];
shared int totalNumVisibleThisTile = 0;

void main() {
    // Initialize the marker array
    uint workGroupSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
    for (uint i = gl_LocalInvocationIndex; i < MAX_TOTAL_VPLS_PER_FRAME; i += workGroupSize) {
        activeLightMarker[i] = 0;
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
        float distance = length(lightPositions[lightIndex].xyz - fragPos);
        float radius = lightRadii[lightIndex];
        if (distance > radius) continue;
        
        int prev = atomicAdd(activeLightMarker[lightIndex], 1);
        // If we're the first to increment for this light, add to visible light set
        if (prev == 0) {
            prev = atomicAdd(totalNumVisibleThisTile, 1);
            // If we exceeded, break and end
            if (prev >= MAX_VPLS_PER_TILE) {
                atomicAdd(totalNumVisibleThisTile, -1);
                break;
            }
            vplIndicesVisiblePerTile[baseTileIndex + prev] = lightIndex;
        }
    }

    // Wait for work group to finish
    barrier();

    // Wait for the group to finish
    barrier();

    // If we're the first in the group update the total number visible
    if (gl_LocalInvocationIndex == 0) {
        vplNumVisiblePerTile[tileCoords.x + tileCoords.y * numTiles.x] = totalNumVisibleThisTile;
    }
}
*/

#define SHUFFLE_DOWN(values, start)                                          \
    for (int index_ = MAX_VPLS_PER_TILE - 2; index_ >= start; --index_) {    \
        values[index_ + 1] = values[index_];                                 \
    }                                                                        \
    values[start] = 0;                              

void main() {
    uvec2 numTiles = uvec2(viewportWidth, viewportHeight);
    uvec2 tileCoords = gl_GlobalInvocationID.xy;
    uvec2 pixelCoords = tileCoords;
    // See https://stackoverflow.com/questions/40574677/how-to-normalize-image-coordinates-for-texture-space-in-opengl
    vec2 texCoords = (vec2(pixelCoords) + vec2(0.5)) / vec2(viewportWidth, viewportHeight);

    int tileIndex = int(tileCoords.x + tileCoords.y * numTiles.x);
    int baseTileIndex = tileIndex * MAX_VPLS_PER_TILE;
    vec3 fragPos = texture(gPosition, texCoords).xyz;

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
        float distance = length(lightPositions[lightIndex].xyz - fragPos);
        float radius = lightRadii[lightIndex];
        float lightIntensity = length(lightColors[lightIndex]);
        float ratio = distance / radius;
        if (ratio > 1.0 || ratio < 0.025) continue;
        //if (ratio > 1.0 || (lightIntensity > 100 && ratio < 0.045) || ratio < 0.02) continue;

        distance = distance * (1.0 - ratio);
        for (int ii = 0; ii < MAX_VPLS_PER_TILE; ++ii) {
            if (distance < distancesVisibleThisTile[ii]) {
                //shuffleDown(indicesVisibleThisTile, ii);
                //shuffleDown(distancesVisibleThisTile, ii);
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