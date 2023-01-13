STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#include "pbr.glsl"

// GBuffer information
uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D gBaseReflectivity;
uniform sampler2D gRoughnessMetallicAmbient;

// Screen space ambient occlusion
uniform sampler2DRect ssao;

// Camera information
uniform vec3 viewPosition;

// Window information
uniform int viewportWidth;
uniform int viewportHeight;

// in/out frame texture
uniform sampler2D screenInput;
layout (binding = 0) writeonly uniform image2D screenOutput;

layout (binding = 21) buffer vplActiveLights {
    int numActiveVPLs;
};

// Active light indices into main buffer
layout (binding = 22) buffer vplIndices {
    int activeLightIndices[];
};

// Resident shadow maps
layout (binding = 23) buffer vplShadowMaps {
    samplerCube shadowCubeMaps[];
};

// Shadow factors for infinite light
layout (std430, binding = 24) buffer vplShadowFactors {
    float shadowFactors[];
};

// Light positions
layout (std430, binding = 25) buffer vplPositions {
    vec4 lightPositions[];
};

// Light colors
layout (std430, binding = 26) buffer vplColors {
    vec4 lightColors[];
};

// Light far planes
layout (std430, binding = 27) buffer vplLightFarPlanes {
    float lightFarPlanes[];
};

#define MAX_VPLS_PER_SCENE 2048
shared vec3 vplPerPixelColorValues[MAX_VPLS_PER_SCENE];

void performLightingCalculations(ivec2 pixelCoords, vec2 texCoords) {
    uint baseLightIndex = gl_LocalInvocationID.z;
    vec3 fragPos = texture(gPosition, texCoords).rgb;
    vec3 viewDir = normalize(viewPosition - fragPos);

    for ( ; baseLightIndex < numActiveVPLs; baseLightIndex += gl_WorkGroupSize.z) {
        // Calculate true light index via lookup into active light table
        int lightIndex = activeLightIndices[baseLightIndex];
        vec3 baseColor = texture(gAlbedo, texCoords).rgb;
        vec3 normal = normalize(texture(gNormal, texCoords).rgb * 2.0 - vec3(1.0));
        float roughness = texture(gRoughnessMetallicAmbient, texCoords).r;
        float metallic = texture(gRoughnessMetallicAmbient, texCoords).g;
        // Note that we take the AO that may have been packed into a texture and augment it by SSAO
        // Note that singe SSAO is sampler2DRect, we need to sample in pixel coordinates and not texel coordinates
        float ambient = texture(gRoughnessMetallicAmbient, texCoords).b * texture(ssao, pixelCoords).r;
        vec3 baseReflectivity = texture(gBaseReflectivity, texCoords).rgb;

        float shadowFactor = calculateShadowValue(shadowCubeMaps[lightIndex], lightFarPlanes[lightIndex], fragPos, lightPositions[lightIndex].xyz, dot(lightPositions[lightIndex].xyz - fragPos, normal), 27);
        // Depending on how visible this VPL is to the infinite light, we want to constrain how bright it's allowed to be
        shadowFactor = lerp(shadowFactor, 0.0, shadowFactors[lightIndex]);

        vplPerPixelColorValues[lightIndex] = calculatePointLighting(fragPos, baseColor, normal, viewDir, lightPositions[lightIndex].xyz, lightColors[lightIndex].xyz, roughness, metallic, ambient, shadowFactor, baseReflectivity);
    }

    // Wait until other threads in the work group have finished
    //barrier();
    memoryBarrierShared();
}

void main() {
    // If we're the first in the local work group, add up all light contributions
    for (uint x = gl_GlobalInvocationID.x; x < viewportWidth; x += gl_NumWorkGroups.x) {
        for (uint y = gl_GlobalInvocationID.y; y < viewportHeight; y += gl_NumWorkGroups.y) {
            ivec2 pixelCoords = ivec2(x, y);
            vec2 texCoords = vec2(pixelCoords) / vec2(viewportWidth, viewportHeight);
            vec3 finalLightColor = texture(screenInput, texCoords).rgb;
            imageStore(screenOutput, pixelCoords, vec4(finalLightColor, 1.0));
        }
    }

    if (gl_LocalInvocationID.z == 0) {

    }
}