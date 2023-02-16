STRATUS_GLSL_VERSION

#include "common.glsl"
#include "atmospheric_postfx.glsl"
#include "pbr.glsl"

uniform sampler2DRect atmosphereBuffer;
uniform vec3 atmosphericLightPos;

#define MAX_LIGHTS 200
// Apple limits us to 16 total samplers active in the pipeline :(
#define MAX_SHADOW_LIGHTS 48
#define SPECULAR_MULTIPLIER 128.0
//#define AMBIENT_INTENSITY 0.00025

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D gBaseReflectivity;
uniform sampler2D gRoughnessMetallicAmbient;
uniform sampler2DRect ssao;

uniform float windowWidth;
uniform float windowHeight;

/**
 * Information about the camera
 */
uniform vec3 viewPosition;

/**
 * Lighting information. All values related
 * to positions should be in world space.
 */
uniform vec3 lightPositions[MAX_LIGHTS];
uniform vec3 lightColors[MAX_LIGHTS];
uniform float lightRadii[MAX_LIGHTS];
uniform bool lightCastsShadows[MAX_LIGHTS];
uniform samplerCube shadowCubeMaps[MAX_SHADOW_LIGHTS];
uniform float lightFarPlanes[MAX_SHADOW_LIGHTS];
//uniform bool lightIsLightProbe[MAX_HAD]
// Since max lights is an upper bound, this can
// tell us how many lights are actually present
uniform int numLights = 0;
uniform int numShadowLights = 0;

/**
 * Information about the directional infinite light (if there is one)
 */
uniform bool infiniteLightingEnabled = false;
uniform vec3 infiniteLightColor;
// uniform float cascadeSplits[4];
// Allows us to take the texture coordinates and convert them to light space texture coordinates for cascade 0
// uniform mat4 cascade0ProjView;

in vec2 fsTexCoords;

layout (location = 0) out vec3 fsColor;

void main() {
    vec2 texCoords = fsTexCoords;
    vec3 fragPos = texture(gPosition, texCoords).rgb;
    vec3 viewDir = normalize(viewPosition - fragPos);

    vec3 baseColor = texture(gAlbedo, texCoords).rgb;
    // Normals generally have values from [-1, 1], but inside
    // an OpenGL texture they are transformed to [0, 1]. To convert
    // them back, we multiply by 2 and subtract 1.
    vec3 normal = normalize(texture(gNormal, texCoords).rgb * 2.0 - vec3(1.0)); // [0, 1] -> [-1, 1]
    float roughness = texture(gRoughnessMetallicAmbient, texCoords).r;
    float metallic = texture(gRoughnessMetallicAmbient, texCoords).g;
    // Note that we take the AO that may have been packed into a texture and augment it by SSAO
    // Note that singe SSAO is sampler2DRect, we need to sample in pixel coordinates and not texel coordinates
    float ambient = texture(gRoughnessMetallicAmbient, texCoords).b * texture(ssao, texCoords * vec2(windowWidth, windowHeight)).r;
    vec3 baseReflectivity = texture(gBaseReflectivity, texCoords).rgb;

    vec3 color = vec3(0.0);
    int shadowIndex = 0;
    for (int i = 0; i < numLights; ++i) {
        //if (i >= numLights) break;

        // Check if we should perform any sort of shadow calculation
        int shadowCubeMapIndex = MAX_LIGHTS;
        float shadowFactor = 0.0;
        if (shadowIndex < numShadowLights && lightCastsShadows[i] == true) {
            // The cube maps are indexed independently from the light index
            shadowCubeMapIndex = shadowIndex;
            ++shadowIndex;
        }

        // calculate distance between light source and current fragment
        float distance = length(lightPositions[i] - fragPos);
        if(distance < lightRadii[i]) {
            if (shadowCubeMapIndex < MAX_LIGHTS) {
                shadowFactor = calculateShadowValue(shadowCubeMaps[shadowCubeMapIndex], lightFarPlanes[shadowCubeMapIndex], fragPos, lightPositions[i], dot(lightPositions[i] - fragPos, normal), 27);
            }
            color = color + calculatePointLighting(fragPos, baseColor, normal, viewDir, lightPositions[i], lightColors[i], roughness, metallic, ambient, shadowFactor, baseReflectivity);
        }
    }

    if (infiniteLightingEnabled) {
        vec3 lightDir = infiniteLightDirection;
        // vec3 cascadeCoord0 = (cascade0ProjView * vec4(fragPos, 1.0)).rgb;
        // cascadeCoord0 = cascadeCoord0 * 0.5 + 0.5;
        vec3 cascadeBlends = vec3(dot(cascadePlanes[0], vec4(fragPos, 1.0)),
                                  dot(cascadePlanes[1], vec4(fragPos, 1.0)),
                                  dot(cascadePlanes[2], vec4(fragPos, 1.0)));
        float shadowFactor = calculateInfiniteShadowValue(vec4(fragPos, 1.0), cascadeBlends, normal);
        //vec3 lightDir = infiniteLightDirection;
        color = color + calculateLighting(infiniteLightColor, lightDir, viewDir, normal, baseColor, roughness, metallic, ambient, shadowFactor, baseReflectivity, 1.0, worldLightAmbientIntensity);
    }
    else {
        color = color + baseColor * ambient * ambientIntensity;
    }

    fsColor = boundHDR(color);
}