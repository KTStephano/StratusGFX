STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "pbr.glsl"
#include "pbr2.glsl"
#include "vpl_common.glsl"

// Input from vertex shader
in vec2 fsTexCoords;

// Outputs
out vec3 color;
out vec3 shadow;

#define STANDARD_MAX_SAMPLES_PER_PIXEL 5
#define ABSOLUTE_MAX_SAMPLES_PER_PIXEL 10
#define MAX_RESAMPLES_PER_PIXEL STANDARD_MAX_SAMPLES_PER_PIXEL

//#define MAX_SHADOW_SAMPLES_PER_PIXEL 25

// GBuffer information
uniform sampler2D gDepth;
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

// Window information
uniform int viewportWidth;
uniform int viewportHeight;

// in/out frame texture
uniform sampler2D screen;

// Accumulated history depth
uniform sampler2D historyDepth;

// Screen tile information
uniform int numTilesX;
uniform int numTilesY;

uniform mat4 invProjectionView;

// Used for random number generation
uniform float time;
uniform int frameCount;

layout (std430, binding = 0) readonly buffer inputBlock1 {
    VplData lightData[];
};

layout (std430, binding = 1) readonly buffer inputBlock2 {
    int numVisible[];
};

uniform samplerCubeArray shadowCubeMaps[MAX_TOTAL_SHADOW_ATLASES];

layout (std430, binding = 3) readonly buffer inputBlock4 {
    AtlasEntry shadowIndices[];
};

layout (std430, binding = 4) readonly buffer inputBlock5 {
    HaltonEntry haltonSequence[];
};

uniform int haltonSize;

void performLightingCalculations(vec3 screenColor, vec2 pixelCoords, vec2 texCoords) {
    //if (length(screenColor) > 0.5) discard;

    // ivec2 numTiles = ivec2(numTilesX, numTilesY);
    // // For example: 16, 9
    // ivec2 multiplier = ivec2(viewportWidth, viewportHeight) / numTiles;
    // uvec2 tileCoords = uvec2(pixelCoords / vec2(multiplier));
    // //uvec2 tileCoords = uvec2(pixelCoords);
    // if (tileCoords.x >= numTiles.x || tileCoords.y >= numTiles.y) return screenColor;

    // // Each entry in the activeLightIndicesPerTile buffer has MAX_VPLS_PER_TILE entries
    // int baseTileIndex = int(tileCoords.x + tileCoords.y * numTiles.x);
    // // numActiveVPLsPerTile only has one int per entry
    // int numActiveVPLs = tileData[baseTileIndex].numVisible;
    // if (numActiveVPLs > MAX_VPLS_PER_TILE) return screenColor;

    float depth = textureLod(gDepth, texCoords, 0).r;
    vec3 fragPos = worldPositionFromDepth(texCoords, depth, invProjectionView);
    //vec3 fragPos = texture(gPosition, texCoords).rgb;
    vec3 viewDir = normalize(viewPosition - fragPos);
    float distToCamera = length(viewPosition - fragPos);

    vec3 baseColor = textureLod(gAlbedo, texCoords, 0).rgb;
    baseColor = vec3(max(baseColor.r, 0.01), max(baseColor.g, 0.01), max(baseColor.b, 0.01));
    //vec3 normalizedBaseColor = baseColor / max(length(baseColor), PREVENT_DIV_BY_ZERO);
    vec3 normal = normalize(textureLod(gNormal, texCoords, 0).rgb * 2.0 - vec3(1.0));
    float roughness = textureLod(gRoughnessMetallicAmbient, texCoords, 0).r;
    roughness = max(0.5, roughness);
    float metallic = textureLod(gRoughnessMetallicAmbient, texCoords, 0).g;
    // Note that we take the AO that may have been packed into a texture and augment it by SSAO
    // Note that singe SSAO is sampler2DRect, we need to sample in pixel coordinates and not texel coordinates
    float ambientOcclusion = clamp(texture(ssao, pixelCoords).r, 0.35, 1.0);
    float ambient = textureLod(gRoughnessMetallicAmbient, texCoords, 0).b;// * ambientOcclusion;
    vec3 baseReflectivity = textureLod(gBaseReflectivity, texCoords, 0).rgb;

    float history = textureLod(historyDepth, texCoords, 0).r;

    vec3 vplColor = vec3(0.0); //screenColor;

    //int maxSamples = numVisible < MAX_SAMPLES_PER_PIXEL ? numVisible : MAX_SAMPLES_PER_PIXEL;
    //int maxShadowLights = numVisible < MAX_SHADOW_SAMPLES_PER_PIXEL ? numVisible : MAX_SHADOW_SAMPLES_PER_PIXEL;

    // Used to seed the pseudo-random number generator
    // See https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
    vec3 seed = vec3(0.0, 0.0, time);
    int maxRandomIndex = numVisible[0] - 1;
    // float shadowFactor = 0.0;
    // for (int baseLightIndex = 0; baseLightIndex < maxShadowLights; ++baseLightIndex) {
    //     //seed.z += 1000.0;
    //     //float rand = random(seed);
    //     //int lightIndex = vplVisibleIndex[int(maxRandomIndex * rand)];
    //     int lightIndex = vplVisibleIndex[baseLightIndex];
    //     vec3 lightPosition = lightData[lightIndex].position.xyz;
    //     AtlasEntry entry = shadowIndices[lightIndex];

    //     shadowFactor += calculateShadowValue1Sample(shadowCubeMaps[entry.index], entry.layer, lightData[lightIndex].farPlane, fragPos, lightPosition, dot(lightPosition - fragPos, normal));
    // }

    // shadowFactor /= float(maxShadowLights);

    const float seedZMultiplier = 10000.0;
    const float seedZOffset = 10000.0;
    //float seedZ = float(frameCount % 20) * seedZMultiplier;
    float seedZ = time;
    seed = vec3(gl_FragCoord.xy, time);
    //seed = vec3(gl_FragCoord.xy, time + float(frameCount));
    //seed = vec3(gl_FragCoord.xy, 0.0);
    //seed = vec3(gl_FragCoord.xy, seedZ);
    //seed = vec3(0.0, 0.0, time);
    //seed = vec3(0.0, gl_FragCoord.y, 0.0);
    //seed = vec3(distToCamera * 10.0, 0.0, time);
    //seed = vec3(distToCamera, 0.0, time);
    //seed = vec3(fragPos.x, fragPos.y, fragPos.z + time);
    //seed = vec3(0.0, gl_FragCoord.y, time);
    //int haltonIndex = int(ceil(haltonSize * random(seed)));
    float validSamples = 0.0;
    bool useBase2 = false;
    //float rand = random(seed);
    //int offset = frameCount % numVisible;
    vec3 colorNoShadow = vec3(0.0);
    //int startIndex = int(gl_FragCoord.x) + int(gl_FragCoord.y) * haltonSize;
    //int startIndex = int(random(seed) * haltonSize);
    //int startIndex = 0;
    //int baseHaltonIndex = int((float(gl_FragCoord.x) / float(haltonSize)) * float(haltonSize)) % haltonSize;
    // float normalizedHistory = 1.0 - history / float(MAX_SAMPLES_PER_PIXEL);
    // //int sampleCount = history < float(MAX_SAMPLES_PER_PIXEL) ? int(normalizedHistory * MAX_SAMPLES_PER_PIXEL) : 1;
    // int sampleCount = int(normalizedHistory * MAX_SAMPLES_PER_PIXEL);
    // sampleCount = max(sampleCount, 1);
    //int sampleCount = history < 2.0 ? MAX_SAMPLES_PER_PIXEL : 1;
    //int sampleCount = history < float(MAX_SAMPLES_PER_PIXEL) ? MAX_SAMPLES_PER_PIXEL : 1;
    //int sampleCount = history < float(MAX_SAMPLES_PER_PIXEL) ? MAX_SAMPLES_PER_PIXEL : HALF_MAX_SAMPLES_PER_PIXEL;
    float distRatioToCamera = min(1.0 - distToCamera / 1000.0, 1.0);
    int samplesMax = history < ABSOLUTE_MAX_SAMPLES_PER_PIXEL ? ABSOLUTE_MAX_SAMPLES_PER_PIXEL : STANDARD_MAX_SAMPLES_PER_PIXEL;
    int sampleCount = max(1, int(samplesMax * distRatioToCamera));
    for (int i = 0, resamples = 0, count = 0; i < sampleCount; i += 1, count += 1) {
        //seed.z += 1000.0;
        float rand = random(seed);
        seed.z += seedZOffset;
        //seed.z += 1000.0;
        // float rand = haltonSequence[haltonIndex].base3;
        // if (useBase2) {
        //     rand = haltonSequence[haltonIndex].base2;
        //     ++haltonIndex;
        //     if (haltonIndex > haltonSize) {
        //         haltonIndex = 0;
        //     }
        //     // seed.z += 1000.0;
        //     // haltonIndex = int(ceil(haltonSize * random(seed)));
        // }
        // useBase2 = !useBase2;

        // Calculate true light index via lookup into active light table
        //int lightIndex = tileData[baseTileIndex].indices[baseLightIndex];
        //int lightIndex = vplVisibleIndex[baseLightIndex];
        int lightIndex = int(maxRandomIndex * rand);
        //int lightIndex = int(int(gl_FragCoord.x) + int(gl_FragCoord.y * numVisible) * rand) % numVisible;
        //int lightIndex = int((fsTexCoords.x + rand) * maxRandomIndex) % numVisible;
        // HaltonEntry baseHEntry = haltonSequence[baseHaltonIndex];
        // HaltonEntry offsetHEntry = haltonSequence[int(rand * (haltonSize - 1))];
        // rand = baseHEntry.base2 + offsetHEntry.base3;
        // if ((startIndex % (2 * haltonSize)) % 2 == 0) {
        //     rand = hentry.base3;
        // }
        //int lightIndex = int(numVisible * rand) % numVisible;
        //int lightIndex = (int(maxRandomIndex * rand) + int(gl_FragCoord.x) * numVisible + int(gl_FragCoord.y)) % numVisible;
        //int lightIndex = (int(maxRandomIndex * rand) + i + resamples) % numVisible;
        //int lightIndex = (int(maxRandomIndex * rand) + offset + count) % numVisible;
        AtlasEntry entry = shadowIndices[lightIndex];
        //if (lightIndex > MAX_TOTAL_VPLS_PER_FRAME) continue;

        vec3 lightPosition = lightData[lightIndex].position.xyz;

        // Make sure the light is in the direction of the plane+normal. If n*(a-p) < 0, the point is on the other side of the plane.
        // If 0 the point is on the plane. If > 0 then the point is on the side of the plane visible along the normal's direction.
        // See https://math.stackexchange.com/questions/1330210/how-to-check-if-a-point-is-in-the-direction-of-the-normal-of-a-plane
        vec3 lightMinusFrag = lightPosition - fragPos;
        float sideCheck = dot(normal, normalize(lightMinusFrag));
        if (resamples < MAX_RESAMPLES_PER_PIXEL && sideCheck < 0.0) {
            ++resamples;
            --i;
            continue;
        }

        float lightRadius = lightData[lightIndex].radius;
        float distance = length(lightMinusFrag);

        if (resamples < MAX_RESAMPLES_PER_PIXEL && distance > lightRadius) {
            ++resamples;
            --i;
            continue;
        }

        float distanceRatio = clamp(distance / lightRadius, 0.0, 1.0);

        vec3 lightColor = lightData[lightIndex].color.xyz;
        //float lightIntensity = length(lightColor);

        validSamples += 1.0;
        //float ratio = distance / lightRadius;
        //if (distance > lightRadii[lightIndex]) continue;

        float shadowFactor = 0.0;
        if (distToCamera < 700) {
            shadowFactor = calculateShadowValue1Sample(shadowCubeMaps[entry.index], entry.layer, lightData[lightIndex].farPlane, fragPos, lightPosition, dot(lightPosition - fragPos, normal));
            shadowFactor = min(shadowFactor, mix(0.95, 1.0, distanceRatio));
            // if (shadowFactor > 0.0 && resamples < MAX_RESAMPLES_PER_PIXEL) {
            //     ++resamples;
            //     --i;
            // }
        }
        // Depending on how visible this VPL is to the infinite light, we want to constrain how bright it's allowed to be
        //shadowFactor = lerp(shadowFactor, 0.0, vpl.shadowFactor);

        vec3 tmpColor = ambientOcclusion * calculateVirtualPointLighting2(fragPos, baseColor, normal, viewDir, lightPosition, lightColor, distToCamera, lightRadius, roughness, metallic, ambient, 0.0, baseReflectivity);
        //colorNoShadow += tmpColor;
        vplColor = vplColor + (1.0 - shadowFactor) * tmpColor;
    }

    validSamples = max(validSamples, 1.0);

    color = baseColor;
    shadow = boundHDR(vplColor / (baseColor * validSamples));
}

void main() {
    performLightingCalculations(textureLod(screen, fsTexCoords, 0).rgb, gl_FragCoord.xy, fsTexCoords);
    // See https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
    // vec3 point = vec3(gl_FragCoord.xy, time);
    // point = vec3(random(point));

    // color = point * 1.0;
}