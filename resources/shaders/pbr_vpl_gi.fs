STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "pbr.glsl"
#include "vpl_tiled_deferred_culling.glsl"

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
//uniform samplerCube shadowCubeMaps[MAX_TOTAL_VPLS_PER_FRAME];
uniform vec3 infiniteLightColor;

layout (std430, binding = 11) readonly buffer shadows {
    samplerCube shadowCubeMaps[];
};

// Window information
uniform int viewportWidth;
uniform int viewportHeight;

// in/out frame texture
uniform sampler2D screen;

// Screen tile information
uniform int numTilesX;
uniform int numTilesY;

layout (std430, binding = 3) readonly buffer vplActiveLights {
    int numActiveLightsPerTile[];
};

// Active light indices into main buffer
layout (std430, binding = 4) readonly buffer vplIndices {
    int activeLightIndicesPerTile[];
};

// Light positions
layout (std430, binding = 5) readonly buffer vplPositions {
    vec4 lightPositions[];
};

layout (std430, binding = 6) readonly buffer vplColors {
    vec4 lightColors[];
};

layout (std430, binding = 7) readonly buffer vplRadii {
    float lightRadii[];
};

layout (std430, binding = 8) readonly buffer vplFarPlanes {
    float lightFarPlanes[];
};

layout (std430, binding = 9) readonly buffer vplNumSamples {
    float lightNumSamples[];
};

vec3 performLightingCalculations(vec3 screenColor, vec2 pixelCoords, vec2 texCoords) {
    ivec2 numTiles = ivec2(numTilesX, numTilesY);
    // For example: 16, 9
    ivec2 multiplier = ivec2(viewportWidth, viewportHeight) / numTiles;
    uvec2 tileCoords = uvec2(pixelCoords / vec2(multiplier));
    //uvec2 tileCoords = uvec2(pixelCoords);
    if (tileCoords.x >= numTiles.x || tileCoords.y >= numTiles.y) return screenColor;

    // Each entry in the activeLightIndicesPerTile buffer has MAX_VPLS_PER_TILE entries
    int baseTileIndex = int(tileCoords.x * MAX_VPLS_PER_TILE + tileCoords.y * numTiles.x * MAX_VPLS_PER_TILE);
    // numActiveVPLsPerTile only has one int per entry
    int numActiveVPLs = numActiveLightsPerTile[tileCoords.x + tileCoords.y * numTiles.x];
    if (numActiveVPLs > MAX_VPLS_PER_TILE) return screenColor;

    vec3 fragPos = texture(gPosition, texCoords).rgb;
    vec3 viewDir = normalize(viewPosition - fragPos);

    vec3 baseColor = texture(gAlbedo, texCoords).rgb;
    vec3 normal = normalize(texture(gNormal, texCoords).rgb * 2.0 - vec3(1.0));
    float roughness = texture(gRoughnessMetallicAmbient, texCoords).r;
    float metallic = texture(gRoughnessMetallicAmbient, texCoords).g;
    // Note that we take the AO that may have been packed into a texture and augment it by SSAO
    // Note that singe SSAO is sampler2DRect, we need to sample in pixel coordinates and not texel coordinates
    float ambient = texture(gRoughnessMetallicAmbient, texCoords).b * texture(ssao, pixelCoords).r;
    vec3 baseReflectivity = texture(gBaseReflectivity, texCoords).rgb;

    vec3 vplColor = screenColor;
    for (int baseLightIndex = 0 ; baseLightIndex < numActiveVPLs; baseLightIndex += 1) {
        // Calculate true light index via lookup into active light table
        int lightIndex = activeLightIndicesPerTile[baseTileIndex + baseLightIndex];
        //if (lightIndex > MAX_TOTAL_VPLS_PER_FRAME) continue;

        vec3 lightPosition = lightPositions[lightIndex].xyz;
        float distanceRadiusRatio = length(lightPosition - fragPos) / lightRadii[lightIndex];
        vec3 lightColor = lightColors[lightIndex].xyz;
        float lightIntensity = length(lightColor);
        //if (distance > lightRadii[lightIndex]) continue;

        int numSamples = 1; //int(lightNumSamples[lightIndex]);
        float shadowFactor = 0.0;
        //if (length(lightPosition - viewPosition) < 150) {
        shadowFactor = calculateShadowValue(shadowCubeMaps[lightIndex], lightFarPlanes[lightIndex], fragPos, lightPosition, dot(lightPosition - fragPos, normal), numSamples);
        //}
        // Depending on how visible this VPL is to the infinite light, we want to constrain how bright it's allowed to be
        //shadowFactor = lerp(shadowFactor, 0.0, vpl.shadowFactor);

        vplColor = vplColor + calculateVirtualPointLighting(fragPos, baseColor, normal, viewDir, lightPosition, lightColor, roughness, metallic, ambient, shadowFactor, baseReflectivity);
    }

    return boundHDR(vplColor);
}

void main() {
    color = performLightingCalculations(texture(screen, fsTexCoords).rgb, gl_FragCoord.xy, fsTexCoords);
}