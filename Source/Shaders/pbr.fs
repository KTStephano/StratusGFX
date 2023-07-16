// Handles the local (point) lighting

STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "common.glsl"
#include "atmospheric_postfx.glsl"
#include "pbr.glsl"
#include "pbr2.glsl"

uniform sampler2DRect atmosphereBuffer;
uniform vec3 atmosphericLightPos;

// This is synchronized with the version in StratusGpuCommon.h
struct PointLight {
    vec4 position;
    vec4 color;
    float radius;
    float farPlane;
    float _1[2];
};

#define SPECULAR_MULTIPLIER 128.0
//#define AMBIENT_INTENSITY 0.00025

uniform sampler2D gDepth;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D gBaseReflectivity;
uniform sampler2D gRoughnessMetallicAmbient;
uniform sampler2DRect ssao;

uniform mat4 invProjectionView;

uniform float windowWidth;
uniform float windowHeight;

/**
 * Information about the camera
 */
uniform vec3 viewPosition;

uniform float emissionStrength = 0.0;

/**
 * Lighting information. All values related
 * to positions should be in world space.
 */
//uniform bool lightIsLightProbe[MAX_HAD]
// Since max lights is an upper bound, this can
// tell us how many lights are actually present
uniform int numLights = 0;
uniform int numShadowLights = 0;

layout (std430, binding = 0) readonly buffer input1 {
    PointLight nonShadowCasters[];
};

uniform samplerCubeArray shadowCubeMaps[MAX_TOTAL_SHADOW_ATLASES];

layout (std430, binding = 1) readonly buffer input2 {
    AtlasEntry shadowIndices[];
};

layout (std430, binding = 2) readonly buffer input3 {
    PointLight shadowCasters[];
};

/**
 * Information about the directional infinite light (if there is one)
 */
uniform vec3 infiniteLightColor;
// uniform float cascadeSplits[4];
// Allows us to take the texture coordinates and convert them to light space texture coordinates for cascade 0
// uniform mat4 cascade0ProjView;

smooth in vec2 fsTexCoords;

layout (location = 0) out vec3 fsColor;

void main() {
    vec2 texCoords = fsTexCoords;
    float depth = textureLod(gDepth, texCoords, 0).r;
    vec3 fragPos = worldPositionFromDepth(texCoords, depth, invProjectionView);
    //vec3 fragPos = texture(gPosition, texCoords).rgb;
    vec3 viewMinusFrag = viewPosition - fragPos;
    vec3 viewDir = normalize(viewMinusFrag);
    float viewDist = length(viewMinusFrag);

    vec4 albedo = textureLod(gAlbedo, texCoords, 0).rgba;
    vec3 baseColor = albedo.rgb;
    // Normals generally have values from [-1, 1], but inside
    // an OpenGL texture they are transformed to [0, 1]. To convert
    // them back, we multiply by 2 and subtract 1.
    vec3 normal = normalize(textureLod(gNormal, texCoords, 0).rgb * 2.0 - vec3(1.0)); // [0, 1] -> [-1, 1]
    vec3 roughnessMetallicEmissive = textureLod(gRoughnessMetallicAmbient, texCoords, 0).rgb;
    float roughness = roughnessMetallicEmissive.r;
    float metallic = roughnessMetallicEmissive.g;
    // Note that we take the AO that may have been packed into a texture and augment it by SSAO
    // Note that singe SSAO is sampler2DRect, we need to sample in pixel coordinates and not texel coordinates
    float ambient = texture(ssao, texCoords * vec2(windowWidth, windowHeight)).r; //textureLod(gRoughnessMetallicAmbient, texCoords, 0).b * texture(ssao, texCoords * vec2(windowWidth, windowHeight)).r;
    vec2 baseReflectivity = textureLod(gBaseReflectivity, texCoords, 0).rg;
    vec3 emissive = vec3(albedo.a, baseReflectivity.g, roughnessMetallicEmissive.b);

    vec3 color = vec3(0.0);
    for (int i = 0; i < numLights; ++i) {
        PointLight light = nonShadowCasters[i];
        // calculate distance between light source and current fragment
        float distance = length(light.position.xyz - fragPos);
        if(distance < light.radius) {
            color = color + calculatePointLighting2(fragPos, baseColor, normal, viewDir, light.position.xyz, light.color.xyz, viewDist, roughness, metallic, ambient, 0, vec3(baseReflectivity.r));
        }
    }

    for (int i = 0; i < numShadowLights; ++i) {
        PointLight light = shadowCasters[i];
        // calculate distance between light source and current fragment
        float distance = length(light.position.xyz - fragPos);
        if(distance < light.radius) {
            float shadowFactor = 0.0;
            AtlasEntry entry = shadowIndices[i];
            if (viewDist < 100.0) {
                shadowFactor = calculateShadowValue8Samples(shadowCubeMaps[entry.index], entry.layer, light.farPlane, fragPos, light.position.xyz, dot(light.position.xyz - fragPos, normal));
            }
            else if (viewDist < 650.0) {
                shadowFactor = calculateShadowValue1Sample(shadowCubeMaps[entry.index], entry.layer, light.farPlane, fragPos, light.position.xyz, dot(light.position.xyz - fragPos, normal));
            }
            color = color + calculatePointLighting2(fragPos, baseColor, normal, viewDir, light.position.xyz, light.color.xyz, viewDist, roughness, metallic, ambient, shadowFactor, vec3(baseReflectivity.r));
        }
    }

#ifdef INFINITE_LIGHTING_ENABLED
    vec3 lightDir = infiniteLightDirection;
    // vec3 cascadeCoord0 = (cascade0ProjView * vec4(fragPos, 1.0)).rgb;
    // cascadeCoord0 = cascadeCoord0 * 0.5 + 0.5;
    vec3 cascadeBlends = vec3(dot(cascadePlanes[0], vec4(fragPos, 1.0)),
                              dot(cascadePlanes[1], vec4(fragPos, 1.0)),
                              dot(cascadePlanes[2], vec4(fragPos, 1.0)));
    float shadowFactor = calculateInfiniteShadowValue(vec4(fragPos, 1.0), cascadeBlends, normal, true);
    //vec3 lightDir = infiniteLightDirection;
    //color = color + calculateLighting(infiniteLightColor, lightDir, viewDir, normal, baseColor, roughness, metallic, ambient, shadowFactor, baseReflectivity, 1.0, 0.003);
    color = color + calculateDirectionalLighting(infiniteLightColor, lightDir, viewDir, normal, baseColor, viewDist, roughness, metallic, ambient, 1.0 - shadowFactor, vec3(baseReflectivity.r), 0.0);
#endif

    fsColor = boundHDR(color + emissive * emissionStrength);
}

// void main() {
//     vec2 texCoords = fsTexCoords;
//     vec3 fragPos = texture(gPosition, texCoords).rgb;
//     vec3 viewMinusFrag = viewPosition - fragPos;
//     vec3 viewDir = normalize(viewMinusFrag);
//     float viewDist = length(viewMinusFrag);

//     vec3 baseColor = texture(gAlbedo, texCoords).rgb;
//     // Normals generally have values from [-1, 1], but inside
//     // an OpenGL texture they are transformed to [0, 1]. To convert
//     // them back, we multiply by 2 and subtract 1.
//     vec3 normal = normalize(texture(gNormal, texCoords).rgb * 2.0 - vec3(1.0)); // [0, 1] -> [-1, 1]
//     float roughness = texture(gRoughnessMetallicAmbient, texCoords).r;
//     float metallic = texture(gRoughnessMetallicAmbient, texCoords).g;
//     // Note that we take the AO that may have been packed into a texture and augment it by SSAO
//     // Note that singe SSAO is sampler2DRect, we need to sample in pixel coordinates and not texel coordinates
//     float ambient = texture(gRoughnessMetallicAmbient, texCoords).b * texture(ssao, texCoords * vec2(windowWidth, windowHeight)).r;
//     vec3 baseReflectivity = texture(gBaseReflectivity, texCoords).rgb;

//     vec3 color = vec3(0.0);
//     int shadowIndex = 0;
//     for (int i = 0; i < numLights; ++i) {
//         //if (i >= numLights) break;

//         // Check if we should perform any sort of shadow calculation
//         int shadowCubeMapIndex = MAX_LIGHTS;
//         float shadowFactor = 0.0;
//         if (shadowIndex < numShadowLights && lightCastsShadows[i] == true) {
//             // The cube maps are indexed independently from the light index
//             shadowCubeMapIndex = shadowIndex;
//             ++shadowIndex;
//         }

//         // calculate distance between light source and current fragment
//         float distance = length(lightPositions[i] - fragPos);
//         if(distance < lightRadii[i]) {
//             if (shadowCubeMapIndex < MAX_LIGHTS) {
//                 if (viewDist < 150.0) {
//                     shadowFactor = calculateShadowValue8Samples(shadowCubeMaps[shadowCubeMapIndex], lightFarPlanes[shadowCubeMapIndex], fragPos, lightPositions[i], dot(lightPositions[i] - fragPos, normal));
//                 }
//                 else if (viewDist < 750.0) {
//                     shadowFactor = calculateShadowValue1Sample(shadowCubeMaps[shadowCubeMapIndex], lightFarPlanes[shadowCubeMapIndex], fragPos, lightPositions[i], dot(lightPositions[i] - fragPos, normal));
//                 }
//             }
//             color = color + calculatePointLighting2(fragPos, baseColor, normal, viewDir, lightPositions[i], lightColors[i], viewDist, roughness, metallic, ambient, shadowFactor, baseReflectivity);
//         }
//     }

//     if (infiniteLightingEnabled) {
//         vec3 lightDir = infiniteLightDirection;
//         // vec3 cascadeCoord0 = (cascade0ProjView * vec4(fragPos, 1.0)).rgb;
//         // cascadeCoord0 = cascadeCoord0 * 0.5 + 0.5;
//         vec3 cascadeBlends = vec3(dot(cascadePlanes[0], vec4(fragPos, 1.0)),
//                                   dot(cascadePlanes[1], vec4(fragPos, 1.0)),
//                                   dot(cascadePlanes[2], vec4(fragPos, 1.0)));
//         float shadowFactor = calculateInfiniteShadowValue(vec4(fragPos, 1.0), cascadeBlends, normal);
//         //vec3 lightDir = infiniteLightDirection;
//         //color = color + calculateLighting(infiniteLightColor, lightDir, viewDir, normal, baseColor, roughness, metallic, ambient, shadowFactor, baseReflectivity, 1.0, 0.003);
//         color = color + calculateDirectionalLighting(infiniteLightColor, lightDir, viewDir, normal, baseColor, viewDist, roughness, metallic, ambient, 1.0 - shadowFactor, baseReflectivity, 0.0);
//     }

//     fsColor = boundHDR(color);
// }