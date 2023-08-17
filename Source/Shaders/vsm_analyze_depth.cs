STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

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

shared vec2 depthTextureSize;
shared ivec2 residencyTableSize;

// const int pixelOffsets[] = int[](
//     -2, -1, 0, 1, 2
// );
const int pixelOffsets[] = int[](
    -1, 0, 1
);
// const int pixelOffsets[] = int[](
//     0
// );

void updateResidencyStatus(in ivec2 pixelCoords) {
    if (pixelCoords.x < 0 || pixelCoords.x >= residencyTableSize.x ||
        pixelCoords.y < 0 || pixelCoords.y >= residencyTableSize.y) {
        
        return;
    }

    uint prev = uint(imageLoad(prevFramePageResidencyTable, pixelCoords).r);
    uint current = imageAtomicExchange(currFramePageResidencyTable, pixelCoords, frameCount);

    if (prev == 0 && current == 0) {
        int original = atomicAdd(numPagesToMakeResident, 1);
        pageIndices[2 * original] = pixelCoords.x;
        pageIndices[2 * original + 1] = pixelCoords.y;
    }
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
    vec2 depthTexCoords = (vec2(xindex, yindex) + vec2(0.5)) / depthTextureSize;
    //vec2 depthTexCoords = vec2(xindex, yindex) / depthTextureSize;

    // Get current depth and convert to world space
    float depth = textureLod(depthTexture, depthTexCoords, 0).r;

    if (depth >= 1.0) return;

    vec3 worldPosition = worldPositionFromDepth(depthTexCoords, depth, invProjectionView);

    // Convert world position to a coordinate from the light's perspective
    vec4 coords = cascadeProjectionView * vec4(worldPosition, 1.0);
    vec2 cascadeTexCoords = coords.xy / coords.w; // Perspective divide
    // Convert from range [-1, 1] to [0, 1]
    cascadeTexCoords.xy = cascadeTexCoords.xy * 0.5 + vec2(0.5);

    vec2 basePixelCoords = cascadeTexCoords * vec2(residencyTableSize - ivec2(1));

    ivec2 basePixelCoordsLower = ivec2(
        floor(basePixelCoords.x),
        floor(basePixelCoords.y)
    );

    ivec2 basePixelCoordsUpper = ivec2(
        ceil(basePixelCoords.x),
        ceil(basePixelCoords.y)
    );

    if (cascadeTexCoords.x >= 0 && cascadeTexCoords.x <= 1 &&
        cascadeTexCoords.y >= 0 && cascadeTexCoords.y <= 1) {

        for (int x = 0; x < 3; ++x) {
            int xoffset = pixelOffsets[x];
            for (int y = 0; y < 3; ++y) {
                int yoffset = pixelOffsets[y];
                ivec2 pixelCoords1 = basePixelCoordsLower + ivec2(xoffset, yoffset);
                ivec2 pixelCoords2 = basePixelCoordsUpper + ivec2(xoffset, yoffset);

                updateResidencyStatus(pixelCoords1);
                updateResidencyStatus(pixelCoords2);
            }
        }
    }
}