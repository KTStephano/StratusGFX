STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

// This is to determine which lights are visible to which parts of the screen
// where the screen has been partitioned into a 2D grid of tiles

// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
// Also see https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction

// Each thread group processes 32 tiles
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#include "vpl_common.glsl"
#include "common.glsl"
#include "pbr.glsl"

uniform vec3 viewPosition;

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

layout (std430, binding = 1) readonly buffer inputBlock2 {
    VplStage1PerTileOutputs stage1Data[];
};

layout (std430, binding = 2) readonly buffer inputBlock3 {
    int numVisible;
};

layout (std430, binding = 3) readonly buffer inputBlock4 {
    int vplVisibleIndex[];
};

layout (std430, binding = 4) writeonly buffer outputBlock2 {
    VplStage2PerTileOutputs vplNumVisiblePerTile[];
};

layout (std430, binding = 11) readonly buffer vplShadows {
    samplerCube shadowCubeMaps[];
};

#define SHUFFLE_DOWN(values, start)                                          \
    for (int index_ = MAX_VPLS_PER_TILE - 2; index_ >= start; --index_) {    \
        values[index_ + 1] = values[index_];                                 \
    }                                                                        \
    values[start] = 0;                            

void main() {
    uvec2 numTiles = gl_NumWorkGroups.xy * gl_WorkGroupSize.xy;
    uvec2 tileCoords = gl_GlobalInvocationID.xy;

    int tileIndex = int(tileCoords.x + tileCoords.y * numTiles.x);
    vec3 fragPos = stage1Data[tileIndex].averageLocalPosition.xyz;
    vec3 normal = stage1Data[tileIndex].averageLocalNormal.xyz;
    float fragDist = length(fragPos - viewPosition);

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
        vec3 lightPosition = lightData[lightIndex].position.xyz;
        // Make sure the light is in the direction of the plane+normal. If n*(a-p) < 0, the point is on the other side of the plane.
        // If 0 the point is on the plane. If > 0 then the point is on the side of the plane visible along the normal's direction.
        // See https://math.stackexchange.com/questions/1330210/how-to-check-if-a-point-is-in-the-direction-of-the-normal-of-a-plane
        vec3 lightMinusFrag = lightPosition - fragPos;
        float sideCheck = dot(normal, lightMinusFrag);
        if (sideCheck < 0) continue;

        float radius = lightData[lightIndex].radius;
        float distance = length(lightMinusFrag);
        float ratio = distance / radius;

        if (ratio > 1.0) continue;
        //if (ratio > 1.0 || ratio < 0.025) continue;
        //if (ratio > 1.0 || (lightIntensity > 100 && ratio < 0.045) || ratio < 0.02) continue;

        //float percentageStrength = max(1.0 - ratio, 0.001);
        distance = ratio;
        for (int ii = 0; ii < MAX_VPLS_PER_TILE; ++ii) {
            if (distance < distancesVisibleThisTile[ii]) {
                // if (ratio < 0.2 && fragDist < 300) {
                // //if (distancesVisibleThisTile[ii] < FLOAT_MAX) {
                // //if (fragDist < 200) {
                //     float shadowFactor = calculateShadowValue1Sample(shadowCubeMaps[lightIndex], radius, fragPos, lightPosition, dot(lightMinusFrag, normal));
                //     // Light can't see current surface
                //     if (shadowFactor > 0.75) break;
                // }

                if (distancesVisibleThisTile[ii] != FLOAT_MAX) {
                    SHUFFLE_DOWN(indicesVisibleThisTile, ii)
                    SHUFFLE_DOWN(distancesVisibleThisTile, ii)
                }
                indicesVisibleThisTile[ii] = lightIndex;
                distancesVisibleThisTile[ii] = distance;

                if (numVisibleThisTile < MAX_VPLS_PER_TILE) {
                    ++numVisibleThisTile;
                }
                break;
            }
        }
    }

    vplNumVisiblePerTile[tileIndex].numVisible = numVisibleThisTile;
    for (int i = 0; i < numVisibleThisTile; ++i) {
        vplNumVisiblePerTile[tileIndex].indices[i] = indicesVisibleThisTile[i];
    }
}