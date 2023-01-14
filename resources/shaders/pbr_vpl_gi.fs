STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "pbr.glsl"

// Input from vertex shader
in vec2 fsTexCoords;
out vec3 color;

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

// Shadow information
#define MAX_LIGHTS 128
uniform samplerCube shadowCubeMaps[MAX_LIGHTS];
uniform vec3 infiniteLightColor;

// Window information
uniform int viewportWidth;
uniform int viewportHeight;

// in/out frame texture
uniform sampler2D screen;

layout (binding = 21) readonly buffer vplActiveLights {
    int numActiveVPLs;
};

// Active light indices into main buffer
layout (binding = 22) readonly buffer vplIndices {
    int activeLightIndices[];
};

// Shadow factors for infinite light
layout (std430, binding = 24) readonly buffer vplShadowFactors {
    float shadowFactors[];
};

// Light positions
layout (std430, binding = 25) readonly buffer vplPositions {
    vec4 lightPositions[];
};

// Light colors
layout (std430, binding = 26) readonly buffer vplColors {
    vec4 lightColors[];
};

// Light far planes
layout (std430, binding = 27) readonly buffer vplLightFarPlanes {
    float lightFarPlanes[];
};

// Light radii
layout (std430, binding = 28) readonly buffer vplLightRadii {
    float lightRadii[];
};

vec3 performLightingCalculations(vec3 screenColor, vec2 pixelCoords, vec2 texCoords) {
    vec3 fragPos = texture(gPosition, texCoords).rgb;
    vec3 viewDir = normalize(viewPosition - fragPos);

    vec3 vplColor = screenColor;
    for (int baseLightIndex = 0 ; baseLightIndex < numActiveVPLs; baseLightIndex += 1) {
        // Calculate true light index via lookup into active light table
        int lightIndex = activeLightIndices[baseLightIndex];
        vec3 lightPosition = lightPositions[lightIndex].xyz;
        float distance = length(lightPosition - fragPos);
        vec3 lightColor = lightColors[lightIndex].xyz;
        if (distance > lightRadii[lightIndex]) continue;
        //if (length(vplColor) > (length(infiniteLightColor) / 25)) break;

        vec3 baseColor = texture(gAlbedo, texCoords).rgb;
        vec3 normal = normalize(texture(gNormal, texCoords).rgb * 2.0 - vec3(1.0));
        float roughness = texture(gRoughnessMetallicAmbient, texCoords).r;
        float metallic = texture(gRoughnessMetallicAmbient, texCoords).g;
        // Note that we take the AO that may have been packed into a texture and augment it by SSAO
        // Note that singe SSAO is sampler2DRect, we need to sample in pixel coordinates and not texel coordinates
        float ambient = texture(gRoughnessMetallicAmbient, texCoords).b * texture(ssao, pixelCoords).r;
        vec3 baseReflectivity = texture(gBaseReflectivity, texCoords).rgb;

        int numSamples = 2;
        float distToCamera = length(lightPosition - viewPosition);
        if (distToCamera < 100) numSamples = 6;

        float shadowFactor = calculateShadowValue(shadowCubeMaps[lightIndex], lightFarPlanes[lightIndex], fragPos, lightPosition, dot(lightPosition - fragPos, normal), numSamples);
        // Depending on how visible this VPL is to the infinite light, we want to constrain how bright it's allowed to be
        shadowFactor = lerp(shadowFactor, 0.0, shadowFactors[lightIndex]);

        vplColor = vplColor + calculatePointLighting(fragPos, baseColor, normal, viewDir, lightPosition, lightColor, roughness, metallic, ambient, shadowFactor, baseReflectivity);
    }

    return boundHDR(vplColor);
}

void main() {
    color = performLightingCalculations(texture(screen, fsTexCoords).rgb, fsTexCoords * vec2(viewportWidth, viewportHeight), fsTexCoords);
}