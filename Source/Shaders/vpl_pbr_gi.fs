STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "pbr.glsl"
#include "pbr2.glsl"
#include "vpl_common.glsl"
#include "bindings.glsl"

// Input from vertex shader
in vec2 fsTexCoords;

// Outputs
//out vec3 color;
out vec4 reservoir;

#define STANDARD_MAX_SAMPLES_PER_PIXEL 1
#define ABSOLUTE_MAX_SAMPLES_PER_PIXEL 1
#define MAX_RESAMPLES_PER_PIXEL 10

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

uniform float minGiOcclusionFactor = 0.95;

layout (std430, binding = VPL_PROBE_DATA_BINDING) readonly buffer inputBlock1 {
    VplData probes[];
};

layout (std430, binding = VPL_PROBE_INDICES_BINDING) readonly buffer inputBlock2 {
    int visibleIndices[];
};

layout (std430, binding = VPL_PROBE_INDEX_COUNTERS_BINDING) readonly buffer inputBlock3 {
    int visibleIndexCounters[];
};

uniform samplerCubeArray positionCubeMaps[MAX_TOTAL_SHADOW_ATLASES];
uniform samplerCubeArray lightingCubeMaps[MAX_TOTAL_SHADOW_ATLASES];
uniform samplerCubeArray shadowCubeMaps[MAX_TOTAL_SHADOW_ATLASES];

layout (std430, binding = 3) readonly buffer inputBlock4 {
    AtlasEntry shadowIndices[];
};

layout (std430, binding = 4) readonly buffer inputBlock5 {
    HaltonEntry haltonSequence[];
};

uniform int haltonSize;

void performLightingCalculations(vec3 screenColor, vec2 pixelCoords, vec2 texCoords) {
    // if (length(screenColor) > 1.0) {
    //     return;
    // }

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

    ivec3 bucketCoords = clampBaseBucketCoords(computeBaseBucketCoords(fragPos));
    int baseBucketIndex = computeBaseBucketIndex(bucketCoords);
    int offsetBucketIndex = computeOffsetBucketIndex(baseBucketIndex);

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
    vec3 baseReflectivity = vec3(textureLod(gBaseReflectivity, texCoords, 0).r);

    float roughnessWeight = 1.0 - roughness;

    float history = textureLod(historyDepth, texCoords, 0).r;

    vec3 vplColor = vec3(0.0); //screenColor;

    //int maxSamples = numVisible < MAX_SAMPLES_PER_PIXEL ? numVisible : MAX_SAMPLES_PER_PIXEL;
    //int maxShadowLights = numVisible < MAX_SHADOW_SAMPLES_PER_PIXEL ? numVisible : MAX_SHADOW_SAMPLES_PER_PIXEL;

    // Used to seed the pseudo-random number generator
    // See https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
    const float seedZMultiplier = 10000.0;
    const float seedZOffset = 10000.0;

    float seedZ = time;
    vec3 seed = vec3(gl_FragCoord.xy, time);
    float validSamples = 0.0;

    vec3 colorNoShadow = vec3(0.0);

    float distRatioToCamera = min(1.0 - distToCamera / 1000.0, 1.0);
    int maxSamplesPerPixel = int(mix(STANDARD_MAX_SAMPLES_PER_PIXEL, ABSOLUTE_MAX_SAMPLES_PER_PIXEL, roughnessWeight));
    //int maxSamplesPerPixel = STANDARD_MAX_SAMPLES_PER_PIXEL;
    int samplesMax = maxSamplesPerPixel; //history < ABSOLUTE_MAX_SAMPLES_PER_PIXEL ? ABSOLUTE_MAX_SAMPLES_PER_PIXEL : maxSamplesPerPixel;
    samplesMax = max(1, int(samplesMax * distRatioToCamera));
    int sampleCount = samplesMax;//max(1, int(samplesMax * 0.5));

    //int maxRandomIndex = visibleIndices[bucketIndex] - 1; //min(numVisible[0] - 1, int((numVisible[0] - 1) * (1.0 / 3.0)));
    //maxRandomIndex = int(maxRandomIndex * mix(1.0, 0.5, distRatioToCamera));
    int maxRandomIndex = visibleIndexCounters[baseBucketIndex] - 1;
    
    if (maxRandomIndex >= 0) {
        for (int i = 0, resamples = 0, count = 0; i < sampleCount; i += 1, count += 1) {
            float rand = random(seed);                                                                                                          
            seed.z += seedZOffset;                                                                                                              
            int probeIndex = visibleIndices[offsetBucketIndex + int(maxRandomIndex * rand)];                                                                                        
            AtlasEntry entry = shadowIndices[probeIndex];                    
            VplData probe = probes[probeIndex];                                                                   
                                                                                                                                                                                                                                                                                    \
            vec3 probePosition = probe.position.xyz;                                                                            
            vec3 rayFromSurfaceToProbe = normalize(probePosition - fragPos);

            vec3 lightPosition = textureLod(positionCubeMaps[entry.index], vec4(rayFromSurfaceToProbe, float(entry.layer)), 0).xyz;
            vec3 specularLightPosition = lightPosition;
                                                                                                                                                
            /* Make sure the light is in the direction of the plane+normal. If n*(a-p) < 0, the point is on the other side of the plane. */     
            /* If 0 the point is on the plane. If > 0 then the point is on the side of the plane visible along the normal's direction.   */     
            /* See https://math.stackexchange.com/questions/1330210/how-to-check-if-a-point-is-in-the-direction-of-the-normal-of-a-plane */     
            vec3 lightMinusFrag = lightPosition - fragPos;                                                                                      
            float probeRadius = 1000.0;                                                                                  
            float distance = length(lightMinusFrag);                                                                                            
                                                                                                                                                
            if (resamples < MAX_RESAMPLES_PER_PIXEL) {                                                                                          
                float sideCheck = dot(normal, normalize(lightMinusFrag));                                                                       
                if (sideCheck < 0.0 || distance > probeRadius) {                                                                                
                    ++resamples;                                                                                                                
                    --i;                                                                                                                        
                    continue;                                                                                                                   
                }                                                                                                                               
            }                                                                                                                                   
                                                                                                                                                
            float distanceRatio = clamp((2.0 * distance) / probeRadius, 0.0, 1.0);                                                              
            float distAttenuation = distanceRatio;                                                                                              
                                                                                                                                                
            vec3 lightColor = textureLod(lightingCubeMaps[entry.index], vec4(rayFromSurfaceToProbe, float(entry.layer)), 0).rgb * 100000.0 * probe.intensityScale;                                                                                 
                                                                                                                                                
            //float shadowFactor =                                                                                                                
            //distToCamera < 700 ? calculateShadowValue1Sample(shadowCubeMaps[entry.index],                                                       
            //                                                 entry.layer,                                                                       
            //                                                 probeRadius,                                                    
            //                                                 fragPos,                                                                           
            //                                                 probePosition,                                                                     
            //                                                 dot(probePosition - fragPos, normal), 0.0)                                        
            //                   : 0.0;      
            float shadowFactor = calculateShadowValue1Sample(shadowCubeMaps[entry.index],                                                       
                                                            entry.layer,                                                                       
                                                            probeRadius,                                                    
                                                            fragPos,                                                                           
                                                            probePosition,                                                                     
                                                            dot(probePosition - fragPos, normal), 0.01);                                                                                                        
            shadowFactor = min(shadowFactor, mix(minGiOcclusionFactor, 1.0, distanceRatio));                                                  
                                                                                                                                                
            float reweightingFactor = 1.0;                                                                                                      
                                                                                                                                                
            //if (shadowFactor > 0.0) {                                                                                                           
            //    reweightingFactor = (1.0 - distAttenuation) * minGiOcclusionFactor + distAttenuation;                                           
            //}                                                                                                                                   
                                                                                                                                                
            validSamples += reweightingFactor;                                                                                                  
                                                                                                                                                
            vec3 tmpColor = ambientOcclusion * calculateVirtualPointLighting2(fragPos, baseColor, normal, viewDir, specularLightPosition,       
                lightPosition, lightColor, distToCamera, probeRadius, roughness, metallic, ambient, 0.0, baseReflectivity                       
            );                                                                                                                                  
            vplColor = vplColor + (1.0 - shadowFactor) * tmpColor;
        }
    }

    validSamples = max(validSamples, 1.0);

    //color = baseColor + PREVENT_DIV_BY_ZERO;//baseColor;
    //reservoir = vec4(boundHDR(vplColor), validSamples);
    reservoir = vec4(max(boundHDR(vplColor), screenColor), validSamples);
}

void main() {
    performLightingCalculations(textureLod(screen, fsTexCoords, 0).rgb, gl_FragCoord.xy, fsTexCoords);
    // See https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
    // vec3 point = vec3(gl_FragCoord.xy, time);
    // point = vec3(random(point));

    // color = point * 1.0;
}