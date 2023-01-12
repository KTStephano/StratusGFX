STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

layout (local_size_x = 1, local_size_y = 1, local_size_z = 64) in;

#include "pbr.glsl"

uniform int numActiveVPLs;

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

// in/out frame texture
layout (rgba16f) uniform image2D screen;

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

void main() {
    ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);
    // We set all the local sizes to 1 so gl_NumWorkGroups.xy contains
    // screen width/height
    vec2 texCoords = vec2(pixelCoords) / vec2(gl_NumWorkGroups.xy);
    vec3 fragPos = texture(gPosition, texCoords).rgb;
    vec3 viewDir = normalize(viewPosition - fragPos);
    // gl_NumWorkGroups.xy contains the screen width/height but
    // gl_NumWorkGroups.z contains one invocation per light
    uint lightIndex = int(gl_LocalInvocationID.z);

    for ( ; lightIndex < numActiveVPLs; lightIndex += gl_WorkGroupSize.z) {
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
    barrier();
    memoryBarrierShared();

    // If we're the first in the local work group, add up all light contributions
    if (gl_LocalInvocationID.z == 0) {
        vec3 finalLightColor = imageLoad(screen, pixelCoords).xyz;
        for (int i = 0; i < numActiveVPLs; ++i) {
            finalLightColor = boundHDR(finalLightColor + vplPerPixelColorValues[i]);
        }

        imageStore(screen, pixelCoords, vec4(finalLightColor, 1.0));
    }
}