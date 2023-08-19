STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

precision highp float;
precision highp int;
precision highp uimage2D;

#include "common.glsl"

layout (local_size_x = 16, local_size_y = 9, local_size_z = 1) in;

uniform mat4 cascadeProjectionView;
uniform mat4 invProjectionView;

layout (r32ui) readonly uniform uimage2D prevFramePageResidencyTable;
layout (r32ui) coherent uniform uimage2D currFramePageResidencyTable;

uniform sampler2D depthTexture;

uniform uint frameCount;

layout (std430, binding = 0) buffer block1 {
    int numPagesToMakeResident;
};

layout (std430, binding = 1) buffer block2 {
    int pageIndices[];
};

layout (std430, binding = 2) buffer block4 {
    int renderPageIndices[];
};

shared vec2 depthTextureSize;
shared ivec2 residencyTableSize;

// const int pixelOffsets[] = int[](
//     -2, -1, 0, 1, 2
// );
// const int pixelOffsets[] = int[](
//     -1, 0, 1
// );
const int pixelOffsets[] = int[](
    0
);

// See https://stackoverflow.com/questions/3417183/modulo-of-negative-numbers
ivec2 wrapIndex(ivec2 value, ivec2 maxValue) {
    return ivec2(mod(mod(value, maxValue) + maxValue, maxValue));
}

void updateResidencyStatus(in ivec2 coords) {
    ivec2 pixelCoords = coords;

    if (pixelCoords.x < 0 || pixelCoords.x >= residencyTableSize.x ||
        pixelCoords.y < 0 || pixelCoords.y >= residencyTableSize.y) {
        
        pixelCoords = wrapIndex(coords, residencyTableSize);
    }

    uint prev = uint(imageLoad(prevFramePageResidencyTable, pixelCoords).r);
    uint current = imageAtomicExchange(currFramePageResidencyTable, pixelCoords, frameCount);
    //uint current = imageAtomicOr(currFramePageResidencyTable, pixelCoords, 1);
    //imageStore(currFramePageResidencyTable, pixelCoords, uvec4(frameCount));

    if (prev == 0 && current == 0) {
        int original = atomicAdd(numPagesToMakeResident, 1);
        pageIndices[2 * original] = pixelCoords.x;
        pageIndices[2 * original + 1] = pixelCoords.y;
    }

    //renderPageIndices[pixelCoords.x + pixelCoords.y * residencyTableSize.x] = current > 0 ? 1 : 0;
}

void main() {
    if (gl_LocalInvocationID == 0) {
        depthTextureSize = textureSize(depthTexture, 0).xy;
        residencyTableSize = imageSize(currFramePageResidencyTable).xy;
    }

    barrier();

    int xindex = int(gl_GlobalInvocationID.x);
    int yindex = int(gl_GlobalInvocationID.y);

    // Depth tex coords
    //vec2 depthTexCoords = (vec2(xindex, yindex) + vec2(0.5)) / depthTextureSize;
    vec2 depthTexCoords = (vec2(xindex, yindex)) / depthTextureSize;

    // Get current depth and convert to world space
    //float depth = textureLod(depthTexture, depthTexCoords, 0).r;
    float depth = texelFetch(depthTexture, ivec2(xindex, yindex), 0).r;

    if (depth >= 1.0) return;

    vec3 worldPosition = worldPositionFromDepth(depthTexCoords, depth, invProjectionView);

    // Convert world position to a coordinate from the light's perspective
    vec4 coords = cascadeProjectionView * vec4(worldPosition, 1.0);
    vec2 cascadeTexCoords = coords.xy / coords.w; // Perspective divide
    // Convert from range [-1, 1] to [0, 1]
    cascadeTexCoords.xy = cascadeTexCoords.xy * 0.5 + vec2(0.5);

    vec2 basePixelCoords = cascadeTexCoords * vec2(residencyTableSize - ivec2(1));

    updateResidencyStatus(ivec2(basePixelCoords));
    updateResidencyStatus(ivec2(basePixelCoords) + ivec2(-1,  0));
    updateResidencyStatus(ivec2(basePixelCoords) + ivec2( 1,  0));
    updateResidencyStatus(ivec2(basePixelCoords) + ivec2( 0, -1));
    updateResidencyStatus(ivec2(basePixelCoords) + ivec2( 0,  1));
    // updateResidencyStatus(ivec2(basePixelCoords) + ivec2(-1,  1));
    // updateResidencyStatus(ivec2(basePixelCoords) + ivec2(-1, -1));
    // updateResidencyStatus(ivec2(basePixelCoords) + ivec2( 1,  1));
    // updateResidencyStatus(ivec2(basePixelCoords) + ivec2( 1, -1));

    // if (basePixelCoords.x == floor(basePixelCoords.x)) {
    //     updateResidencyStatus(ivec2(basePixelCoords) + ivec2(-1, 0));
    //     updateResidencyStatus(ivec2(basePixelCoords) + ivec2(-1, -1));
    //     updateResidencyStatus(ivec2(basePixelCoords) + ivec2(-1, 1));
    // }

    // if (basePixelCoords.x == ceil(basePixelCoords.x)) {
    //     updateResidencyStatus(ivec2(basePixelCoords) + ivec2(1, 0));
    //     updateResidencyStatus(ivec2(basePixelCoords) + ivec2(1, -1));
    //     updateResidencyStatus(ivec2(basePixelCoords) + ivec2(1, 1));
    // }

    // if (basePixelCoords.y == floor(basePixelCoords.y)) {
    //     updateResidencyStatus(ivec2(basePixelCoords) + ivec2(0, -1));
    //     updateResidencyStatus(ivec2(basePixelCoords) + ivec2(-1, -1));
    //     updateResidencyStatus(ivec2(basePixelCoords) + ivec2(1, -1));
    // }

    // if (basePixelCoords.y == ceil(basePixelCoords.y)) {
    //     updateResidencyStatus(ivec2(basePixelCoords) + ivec2(0, 1));
    //     updateResidencyStatus(ivec2(basePixelCoords) + ivec2(-1, 1));
    //     updateResidencyStatus(ivec2(basePixelCoords) + ivec2(1, 1));
    // }

    // ivec2 basePixelCoordsLower = ivec2(
    //     floor(basePixelCoords.x),
    //     floor(basePixelCoords.y)
    // );

    // ivec2 basePixelCoordsUpper = ivec2(
    //     ceil(basePixelCoords.x),
    //     ceil(basePixelCoords.y)
    // );

    // if (cascadeTexCoords.x >= 0 && cascadeTexCoords.x <= 1 &&
    //     cascadeTexCoords.y >= 0 && cascadeTexCoords.y <= 1) {

    //     for (int x = 0; x < 1; ++x) {
    //         int xoffset = pixelOffsets[x];
    //         for (int y = 0; y < 1; ++y) {
    //             int yoffset = pixelOffsets[y];
    //             ivec2 pixelCoords1 = basePixelCoordsLower + ivec2(xoffset, yoffset);
    //             //ivec2 pixelCoords2 = basePixelCoordsUpper + ivec2(xoffset, yoffset);

    //             updateResidencyStatus(pixelCoords1);
    //             // if (pixelCoords1 != pixelCoords2) {
    //             //     updateResidencyStatus(pixelCoords2);
    //             // }

    //             // ivec2 pixelCoords = pixelCoords1 + ivec2(xoffset, yoffset - 1);
    //             // updateResidencyStatus(pixelCoords);

    //             // pixelCoords = pixelCoords1 + ivec2(xoffset, yoffset + 1);
    //             // updateResidencyStatus(pixelCoords);

    //             // ivec2 pixelCoords1 = basePixelCoordsLower + ivec2(xoffset, yoffset);
    //             // ivec2 pixelCoords2 = basePixelCoordsUpper + ivec2(xoffset, yoffset);
    //             // ivec2 pixelCoords3 = basePixelCoordsLower + ivec2(xoffset - 1, yoffset);
    //             // ivec2 pixelCoords4 = basePixelCoordsUpper + ivec2(xoffset + 1, yoffset);
    //             // ivec2 pixelCoords5 = basePixelCoordsLower + ivec2(xoffset, yoffset - 1);
    //             // ivec2 pixelCoords6 = basePixelCoordsUpper + ivec2(xoffset, yoffset + 1);

    //             // updateResidencyStatus(pixelCoords1);
    //             // updateResidencyStatus(pixelCoords2);
    //             // updateResidencyStatus(pixelCoords3);
    //             // updateResidencyStatus(pixelCoords4);
    //             // updateResidencyStatus(pixelCoords5);
    //             // updateResidencyStatus(pixelCoords6);
    //         }
    //     }
    // }
}