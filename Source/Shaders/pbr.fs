// Handles the local (point) lighting

STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

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
uniform sampler2D ids;

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

layout (std430, binding = POINT_LIGHT_NON_SHADOW_CASTER_BINDING_POINT) readonly buffer input1 {
    PointLight nonShadowCasters[];
};

uniform samplerCubeArray shadowCubeMaps[MAX_TOTAL_SHADOW_ATLASES];

layout (std430, binding = POINT_LIGHT_SHADOW_ATLAS_INDICES_BINDING_POINT) readonly buffer input2 {
    AtlasEntry shadowIndices[];
};

layout (std430, binding = POINT_LIGHT_SHADOW_CASTER_BINDING_POINT) readonly buffer input3 {
    PointLight shadowCasters[];
};

layout (std430, binding = VSM_CURR_FRAME_RESIDENCY_TABLE_BINDING) readonly buffer inputBlock3 {
    PageResidencyEntry currFramePageResidencyTable[];
};

uniform int numPagesXY;

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
    vec2 roughnessMetallic = textureLod(gRoughnessMetallicAmbient, texCoords, 0).rg;
    float roughness = roughnessMetallic.r;
    float metallic = roughnessMetallic.g;
    // Note that we take the AO that may have been packed into a texture and augment it by SSAO
    // Note that singe SSAO is sampler2DRect, we need to sample in pixel coordinates and not texel coordinates
    float ambient = texture(ssao, texCoords * vec2(windowWidth, windowHeight)).r; //textureLod(gRoughnessMetallicAmbient, texCoords, 0).b * texture(ssao, texCoords * vec2(windowWidth, windowHeight)).r;
    float baseReflectivity = textureLod(gBaseReflectivity, texCoords, 0).r;
    vec3 emissive = albedo.a > 0.0 ? albedo.rgb : vec3(0.0);

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
            if (viewDist < 150.0) {
                shadowFactor = calculateShadowValue8Samples(shadowCubeMaps[entry.index], entry.layer, light.farPlane, fragPos, light.position.xyz, dot(light.position.xyz - fragPos, normal), 0.03);
            }
            else if (viewDist < 750.0) {
                shadowFactor = calculateShadowValue1Sample(shadowCubeMaps[entry.index], entry.layer, light.farPlane, fragPos, light.position.xyz, dot(light.position.xyz - fragPos, normal), 0.03);
            }
            color = color + calculatePointLighting2(fragPos, baseColor, normal, viewDir, light.position.xyz, light.color.xyz, viewDist, roughness, metallic, ambient, shadowFactor, vec3(baseReflectivity.r));
        }
    }

#ifdef INFINITE_LIGHTING_ENABLED
    vec3 lightDir = infiniteLightDirection;

    vec3 cascadeBlends = vec3(dot(cascadePlanes[0], vec4(fragPos, 1.0)),
                              dot(cascadePlanes[1], vec4(fragPos, 1.0)),
                              dot(cascadePlanes[2], vec4(fragPos, 1.0)));
    float shadowFactor = calculateInfiniteShadowValue(vec4(fragPos, 1.0), cascadeBlends, normal, true);

    vec3 cacheColor = vec3(0.0);

    // //shadowFactor = 1.0;
    // int cascadeIndex = vsmCalculateCascadeIndexFromWorldPos(fragPos);

    // vec2 pageCoords = vec2(0.0);
    // vec3 cascadeTexCoords = vsmCalculateOriginClipValueFromWorldPos(fragPos, cascadeIndex);
    // pageCoords = cascadeTexCoords.xy * 0.5 + vec2(0.5);
    // pageCoords = pageCoords * vec2(numPagesXY - 1);

    // ivec2 pageCoordsLower = ivec2(floor(pageCoords));
    // ivec2 pageCoordsUpper = ivec2(ceil(pageCoords));

    // uint pageId;
    // uint dirtyBit;

    // int pageFlatIndex = pageCoordsLower.x + pageCoordsLower.y  * int(numPagesXY) + cascadeIndex * int(numPagesXY * numPagesXY);
    // unpackPageIdAndDirtyBit(currFramePageResidencyTable[pageFlatIndex].info, pageId, dirtyBit);
    // //unpackPageIdAndDirtyBit(currFramePageResidencyTable[pageCoordsLower.x + pageCoordsUpper.y * int(numPagesXY)].info, pageId, dirtyBit);

    // vec3 color1 = vec3(0.0, 1.0, 0.0);
    // vec3 color2 = vec3(0.0, 0.0, 1.0);
    // vec3 color3 = vec3(1.0, 1.0, 0.0);
    // vec3 color4 = vec3(0.0, 1.0, 1.0);

    // vec3 colors[] = vec3[](
    //     color1,
    //     color2,
    //     color3,
    //     color4
    // );

    // float percentage = float(cascadeIndex) / float(vsmNumCascades - 1);

    // //vec3 colorMix = mix(color1, color2, percentage);
    // vec3 colorMix = colors[cascadeIndex];

    // //cacheColor = dirtyBit > 0 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 15.0 / 255.0) / 64.0;
    // cacheColor = dirtyBit > 0 ? vec3(1.0, 0.0, 0.0) : colorMix / 16.0;
    // //cacheColor = dirtyBit > 0 ? vec3(1.0, 0.0, 0.0) : vec3(6.0 / 255.0, 86.0 / 255.0, 1.0);

    // if (fract(pageCoords.x) <= 0.02 || fract(pageCoords.x) >= 0.98 ||
    //     fract(pageCoords.y) <= 0.02 || fract(pageCoords.y) >= 0.98) {

    //     //cacheColor = (vec3(1.0, 198.0 / 255.0, 0.0)) * 2.0;
    //     //cacheColor = vec3(1.0, 161.0 / 255.0, 0) * 2.0;
    //     //cacheColor = vec3(6.0 / 255.0, 86.0 / 255.0, 1.0);
    //     //cacheColor = vec3(0, 218.0 / 255.0, 23.0 / 255.0);
    //     cacheColor = colorMix;//vec3(0.0, 1.0, 0.0);
    // }

    color = color + cacheColor + calculateDirectionalLighting(infiniteLightColor, lightDir, viewDir, normal, baseColor, viewDist, roughness, metallic, ambient, 1.0 - shadowFactor, vec3(baseReflectivity.r), 0.0);
#endif

    fsColor = boundHDR(color + emissive * emissionStrength);

    //float currId = texture(ids, fsTexCoords).r;
    //fsColor = vec3(random(currId), random(currId + 1), random(currId + 2));
}