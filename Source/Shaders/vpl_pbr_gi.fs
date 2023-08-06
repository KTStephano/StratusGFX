STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "pbr.glsl"
#include "pbr2.glsl"
#include "vpl_common.glsl"
#include "random.glsl"

// Input from vertex shader
in vec2 fsTexCoords;

// Outputs
out vec3 color;
out vec4 reservoir;

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

uniform float minGiOcclusionFactor = 0.95;

layout (std430, binding = 0) readonly buffer inputBlock1 {
    VplData lightData[];
};

layout (std430, binding = 1) readonly buffer inputBlock2 {
    int numVisible[];
};

layout (std430, binding = 3) readonly buffer inputBlock4 {
    AtlasEntry shadowIndices[];
};

layout (std430, binding = 4) readonly buffer inputBlock5 {
    HaltonEntry haltonSequence[];
};

uniform int haltonSize;

vec3 computeReflection(vec3 v, vec3 normal) {
    return v - 2.0 * dot(v, normal) * normal;
}

void trace(
    inout vec3 seed, 
    in vec3 baseColor,
    in vec3 normal,
    in vec3 fragPos,
    in vec2 roughnessMetallic,
    in vec3 baseReflectivity,
    in vec3 startDirection,
    inout int resamples,
    inout float validSamples,
    inout vec4 traceReservoir) {

    const float seedZMultiplier = 10000.0;
    const float seedZOffset = 10000.0;

    const int maxResamples = 500;

    int maxRandomIndex = numVisible[0] - 1;

    vec3 currDiffuse = baseColor;
    vec3 currFragPos = fragPos;
    vec3 currNormal = normal;
    vec2 currRoughnessMetallic = roughnessMetallic;
    vec3 currDirection = startDirection;

    vec3 lightMask = vec3(0.0);
    //validSamples += 1;

    // Each successful iteration = 1 bounce of light
    for (int i = 0; i < 1 && resamples < maxResamples; i += 1) {
        // if (i == 0) {
        //     validSamples += 1;
        // }
        float rand = random(seed);                                                                                                                                                                                                                      
        int lightIndex = int(maxRandomIndex * rand);                                                                                        
        AtlasEntry entry = shadowIndices[lightIndex];                                                                                       
                                                                                                                                                                                                                                                                                \
        vec3 lightPosition = lightData[lightIndex].position.xyz;                                                                                                                                    
                                                                                                                                            
        /* Make sure the light is in the direction of the plane+normal. If n*(a-p) < 0, the point is on the other side of the plane. */     
        /* If 0 the point is on the plane. If > 0 then the point is on the side of the plane visible along the normal's direction.   */     
        /* See https://math.stackexchange.com/questions/1330210/how-to-check-if-a-point-is-in-the-direction-of-the-normal-of-a-plane */     
        vec3 direction = lightPosition - currFragPos;                                                                                  
        float lightRadius = lightData[lightIndex].radius * 0.25;                                                                                  
        float distance = length(direction);        
        direction = normalize(direction);

        float sideCheck = dot(currNormal, direction);                                                                       
        if (sideCheck < 0.0 || distance > lightRadius) {                                                                                
            ++resamples;                                                                                                                
            --i;                                                                                                                        
            continue;                                                                                                                   
        }                                                        

        const float minBias = 0.05;                                                                                                                                    
        float shadowFactor = calculateShadowValue1Sample(probeTextures[entry.index].occlusion,                                                       
                                                        entry.layer,                                                                       
                                                        lightData[lightIndex].radius,                                                    
                                                        currFragPos,                                                                           
                                                        lightPosition,                                                                     
                                                        dot(direction, currNormal), minBias);
        
        if (shadowFactor > 0.0) {
            ++resamples;                                                                                                                
            --i;                                                                                                                        
            continue; 
        }

        //direction = normalize(direction);

        // TODO: Replace with actual unit sphere random
        //vec3 unit = currRoughnessMetallic.r * randomVector(seed, -1, 1);
        vec3 unit = currRoughnessMetallic.r * randomVector(seed, -1, 1);

        vec3 scatteredVec = normalize(currNormal + unit);
        vec3 reflectedVec = normalize(computeReflection(-direction, currNormal) + unit);
        vec3 target = mix(scatteredVec, reflectedVec, currRoughnessMetallic.g);

        vec4 newDiffuse = textureLod(probeTextures[entry.index].diffuse, vec4(target, float(entry.layer)), 0).rgba;

        if (newDiffuse.a < 0.5) {
            // if (dot(infiniteLightDirection, target) >= 0.906) {
            //     // ++resamples;
            //     // --i;
            //     // continue;
            //     //lightMask = 0.10 * newDiffuse.rgb;
            //     // break;
            //     lightMask = infiniteLightColor;
            //     break;
            // }

            // lightMask = 0.25 * newDiffuse.rgb;
            // currDirection = target;
            // break;
            
            ++resamples;
            --i;
            continue;

            // foundLight = 1.0;
            // //validSamples += 1.0;
            // break;

            // lightMask = 0.5 * infiniteLightColor;
            // break;

            // lightMask = newDiffuse.rgb;
            // break;
            // ++resamples;
            // --i;
            // continue;
        }

        // if (lightData[lightIndex].visible > 0) {
        //     lightMask = infiniteLightColor;
        //     break;
        // }

        float magnitude = lightData[lightIndex].radius * textureLod(probeTextures[entry.index].occlusion, vec4(target, float(entry.layer)), 0).r;
        vec3 newPosition = lightData[lightIndex].position.xyz + 0.99 * magnitude * target;
        vec4 newNormal = textureLod(probeTextures[entry.index].normals, vec4(target, float(entry.layer)), 0).rgba;
        newNormal = vec4(normalize(newNormal.rgb * 2.0 - vec3(1.0)), newNormal.a);
        //float newMetallic = textureLod(probeTextures[entry.index].properties, vec4(target, float(entry.layer)), 0).r;

        vec3 cascadeBlends = vec3(dot(cascadePlanes[0], vec4(newPosition, 1.0)),
                                  dot(cascadePlanes[1], vec4(newPosition, 1.0)),
                                  dot(cascadePlanes[2], vec4(newPosition, 1.0)));
        shadowFactor = calculateInfiniteShadowValue(vec4(newPosition, 1.0), cascadeBlends, newNormal.rgb, true, 0.0);

        if (shadowFactor > 0.0) {
            currDiffuse = currDiffuse * newDiffuse.rgb;
            currFragPos = newPosition;
            currNormal = newNormal.rgb;
            //currRoughnessMetallic = vec2(newNormal.a, newMetallic);
            currDirection = target;
            lightMask = vec3(1.0);
            break;
        }

        ++resamples;
        --i;
    }

    validSamples += 1.0;
    validSamples = max(validSamples, 1.0);

    vec3 viewMinusFrag = viewPosition - fragPos;
    vec3 viewDir = normalize(viewMinusFrag);
    float viewDist = length(viewMinusFrag);

    vec3 tempColor = lightMask * vec3(currDiffuse);
        //return calculateLighting_Lambert(lightColor, lightDir, viewDir, normal, baseColor, viewDist, 0.0, roughness, metallic, ambientOcclusion, adjustedShadowFactor, baseReflectance, vplAttenuation(lightDir, lightRadius), 0.0, vec3(1.0), vec3(1.0));

    tempColor = calculateLighting_Lambert(infiniteLightColor, currDirection, viewDir, normal, tempColor, viewDist, 0.0, roughnessMetallic.r, roughnessMetallic.g, 1.0, length(lightMask), baseReflectivity, 1.0, 0.0, vec3(1.0), vec3(1.0));

    traceReservoir += vec4(tempColor, validSamples);

    traceReservoir = vec4(boundHDR(traceReservoir.rgb), traceReservoir.a);
}

void performLightingCalculations(vec3 screenColor, vec2 pixelCoords, vec2 texCoords) {
    // if (screenColor.r > 0.0 || screenColor.g > 0.0 || screenColor.b > 0.0) {
    //     color = vec3(0.0);
    //     reservoir = vec4(1.0);
    //     return;
    // }

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
    //roughness = max(0.5, roughness);
    float metallic = textureLod(gRoughnessMetallicAmbient, texCoords, 0).g;
    // Note that we take the AO that may have been packed into a texture and augment it by SSAO
    // Note that singe SSAO is sampler2DRect, we need to sample in pixel coordinates and not texel coordinates
    float ambientOcclusion = clamp(texture(ssao, pixelCoords).r, 0.35, 1.0);
    float ambient = textureLod(gRoughnessMetallicAmbient, texCoords, 0).b;// * ambientOcclusion;
    vec3 baseReflectivity = vec3(textureLod(gBaseReflectivity, texCoords, 0).r);

    float roughnessWeight = 1.0 - roughness;

    float history = textureLod(historyDepth, texCoords, 0).r;

    vec3 traceColor = vec3(0.0); //screenColor;

    //int maxSamples = numVisible < MAX_SAMPLES_PER_PIXEL ? numVisible : MAX_SAMPLES_PER_PIXEL;
    //int maxShadowLights = numVisible < MAX_SHADOW_SAMPLES_PER_PIXEL ? numVisible : MAX_SHADOW_SAMPLES_PER_PIXEL;

    // Used to seed the pseudo-random number generator
    // See https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl

    float seedZ = time;
    //vec3 seed = vec3(gl_FragCoord.xy, time);
    vec3 seed = vec3(fragPos.xy, fragPos.z + time);
    float validSamples = 0.0;

    vec3 colorNoShadow = vec3(0.0);

    float distRatioToCamera = min(1.0 - distToCamera / 1000.0, 1.0);
    int maxSamplesPerPixel = int(mix(STANDARD_MAX_SAMPLES_PER_PIXEL, ABSOLUTE_MAX_SAMPLES_PER_PIXEL, roughnessWeight));
    //int maxSamplesPerPixel = STANDARD_MAX_SAMPLES_PER_PIXEL;
    int samplesMax = history < ABSOLUTE_MAX_SAMPLES_PER_PIXEL ? ABSOLUTE_MAX_SAMPLES_PER_PIXEL : maxSamplesPerPixel;
    samplesMax = max(1, int(samplesMax * distRatioToCamera));
    int sampleCount = samplesMax;//max(1, int(samplesMax * 0.5));

    int maxRandomIndex = numVisible[0] - 1; //min(numVisible[0] - 1, int((numVisible[0] - 1) * (1.0 / 3.0)));

    int resamples = 0;
    vec4 traceReservoir = vec4(0.0);

    int numTraceSamples = 1;
    vec3 startDirection = normalize(fragPos - viewPosition);
    for (int i = 0; i < numTraceSamples; ++i) {
    trace(seed, vec3(1.0), normal, fragPos, vec2(roughness, metallic), baseReflectivity, startDirection, resamples, validSamples, traceReservoir);
    }

    color = baseColor;
    reservoir = traceReservoir;// / float(numTraceSamples);

    //maxRandomIndex = int(maxRandomIndex * mix(1.0, 0.5, distRatioToCamera));
    
    // for (int i = 0, resamples = 0, count = 0; i < sampleCount; i += 1, count += 1) {
    //     float rand = random(seed);                                                                                                          
    //     seed.z += seedZOffset;                                                                                                              
    //     int lightIndex = int(maxRandomIndex * rand);                                                                                        
    //     AtlasEntry entry = shadowIndices[lightIndex];                                                                                       
    //                                                                                                                                                                                                                                                                             \
    //     vec3 lightPosition = lightData[lightIndex].position.xyz;                                                                            
    //     vec3 specularLightPosition = lightData[lightIndex].position.xyz;                                                            
                                                                                                                                            
    //     /* Make sure the light is in the direction of the plane+normal. If n*(a-p) < 0, the point is on the other side of the plane. */     
    //     /* If 0 the point is on the plane. If > 0 then the point is on the side of the plane visible along the normal's direction.   */     
    //     /* See https://math.stackexchange.com/questions/1330210/how-to-check-if-a-point-is-in-the-direction-of-the-normal-of-a-plane */     
    //     vec3 lightMinusFrag = lightPosition - fragPos;                                                                                      
    //     float lightRadius = lightData[lightIndex].radius;                                                                                   
    //     float distance = length(lightMinusFrag);                                                                                            
                                                                                                                                            
    //     if (resamples < MAX_RESAMPLES_PER_PIXEL) {                                                                                          
    //         float sideCheck = dot(normal, normalize(lightMinusFrag));                                                                       
    //         if (sideCheck < 0.0 || distance > lightRadius) {                                                                                
    //             ++resamples;                                                                                                                
    //             --i;                                                                                                                        
    //             continue;                                                                                                                   
    //         }                                                                                                                               
    //     }                                                                                                                                   
                                                                                                                                            
    //     float distanceRatio = clamp((2.0 * distance) / lightRadius, 0.0, 1.0);                                                              
    //     float distAttenuation = distanceRatio;                                                                                              
                                                                                                                                            
    //     vec3 lightColor = vec3(50000.0);                                                                                 
                                                                                                                                            
    //     float shadowFactor =                                                                                                                
    //     distToCamera < 700 ? calculateShadowValue1Sample(probeTextures[entry.index].occlusion,                                                       
    //                                                      entry.layer,                                                                       
    //                                                      lightData[lightIndex].radius,                                                    
    //                                                      fragPos,                                                                           
    //                                                      lightPosition,                                                                     
    //                                                      dot(lightPosition - fragPos, normal), 0.05)                                        
    //                        : 0.0;                                                                                                           
    //     shadowFactor = min(shadowFactor, mix(minGiOcclusionFactor, 1.0, distanceRatio));                                                    
                                                                                                                                            
    //     float reweightingFactor = 1.0;                                                                                                      
                                                                                                                                            
    //     if (shadowFactor > 0.0) {                                                                                                           
    //         reweightingFactor = (1.0 - distAttenuation) * minGiOcclusionFactor + distAttenuation;                                           
    //     }                                                                                                                                   
                                                                                                                                            
    //     validSamples += reweightingFactor;                                                                                                  
                                                                                                                                            
    //     vec3 tmpColor = ambientOcclusion * calculateVirtualPointLighting2(fragPos, baseColor, normal, viewDir, specularLightPosition,       
    //         lightPosition, lightColor, distToCamera, lightRadius, roughness, metallic, ambient, 0.0, baseReflectivity                       
    //     );                                                                                                                                  
    //     vplColor = vplColor + (1.0 - shadowFactor) * tmpColor;
    // }
}

// void performLightingCalculations(vec3 screenColor, vec2 pixelCoords, vec2 texCoords) {
//     //if (length(screenColor) > 0.0) return screenColor;

//     float depth = textureLod(gDepth, texCoords, 0).r;
//     vec3 fragPos = worldPositionFromDepth(texCoords, depth, invProjectionView);
//     //vec3 fragPos = texture(gPosition, texCoords).rgb;
//     vec3 viewDir = normalize(viewPosition - fragPos);
//     float distToCamera = length(viewPosition - fragPos);

//     vec3 baseColor = textureLod(gAlbedo, texCoords, 0).rgb;
//     baseColor = vec3(max(baseColor.r, 0.01), max(baseColor.g, 0.01), max(baseColor.b, 0.01));
//     //vec3 normalizedBaseColor = baseColor / max(length(baseColor), PREVENT_DIV_BY_ZERO);
//     vec3 normal = normalize(textureLod(gNormal, texCoords, 0).rgb * 2.0 - vec3(1.0));
//     float roughness = textureLod(gRoughnessMetallicAmbient, texCoords, 0).r;
//     roughness = max(0.5, roughness);
//     float metallic = textureLod(gRoughnessMetallicAmbient, texCoords, 0).g;
//     // Note that we take the AO that may have been packed into a texture and augment it by SSAO
//     // Note that singe SSAO is sampler2DRect, we need to sample in pixel coordinates and not texel coordinates
//     float ambientOcclusion = clamp(texture(ssao, pixelCoords).r, 0.35, 1.0);
//     float ambient = textureLod(gRoughnessMetallicAmbient, texCoords, 0).b;// * ambientOcclusion;
//     vec3 baseReflectivity = vec3(textureLod(gBaseReflectivity, texCoords, 0).r);

//     float roughnessWeight = 1.0 - roughness;

//     float history = textureLod(historyDepth, texCoords, 0).r;

//     vec3 vplColor = vec3(0.0); //screenColor;

//     //int maxSamples = numVisible < MAX_SAMPLES_PER_PIXEL ? numVisible : MAX_SAMPLES_PER_PIXEL;
//     //int maxShadowLights = numVisible < MAX_SHADOW_SAMPLES_PER_PIXEL ? numVisible : MAX_SHADOW_SAMPLES_PER_PIXEL;

//     // Used to seed the pseudo-random number generator
//     // See https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
//     const float seedZMultiplier = 10000.0;
//     const float seedZOffset = 10000.0;

//     float seedZ = time;
//     vec3 seed = vec3(gl_FragCoord.xy, time);
//     float validSamples = 0.0;

//     vec3 colorNoShadow = vec3(0.0);

//     float distRatioToCamera = min(1.0 - distToCamera / 1000.0, 1.0);
//     int maxSamplesPerPixel = int(mix(STANDARD_MAX_SAMPLES_PER_PIXEL, ABSOLUTE_MAX_SAMPLES_PER_PIXEL, roughnessWeight));
//     //int maxSamplesPerPixel = STANDARD_MAX_SAMPLES_PER_PIXEL;
//     int samplesMax = history < ABSOLUTE_MAX_SAMPLES_PER_PIXEL ? ABSOLUTE_MAX_SAMPLES_PER_PIXEL : maxSamplesPerPixel;
//     samplesMax = max(1, int(samplesMax * distRatioToCamera));
//     int sampleCount = samplesMax;//max(1, int(samplesMax * 0.5));

//     int maxRandomIndex = numVisible[0] - 1; //min(numVisible[0] - 1, int((numVisible[0] - 1) * (1.0 / 3.0)));
//     //maxRandomIndex = int(maxRandomIndex * mix(1.0, 0.5, distRatioToCamera));
    
//     for (int i = 0, resamples = 0, count = 0; i < sampleCount; i += 1, count += 1) {
//         float rand = random(seed);                                                                                                          
//         seed.z += seedZOffset;                                                                                                              
//         int lightIndex = int(maxRandomIndex * rand);                                                                                        
//         AtlasEntry entry = shadowIndices[lightIndex];                                                                                       
//                                                                                                                                                                                                                                                                                 \
//         vec3 lightPosition = lightData[lightIndex].position.xyz;                                                                            
//         vec3 specularLightPosition = lightData[lightIndex].position.xyz;                                                            
                                                                                                                                            
//         /* Make sure the light is in the direction of the plane+normal. If n*(a-p) < 0, the point is on the other side of the plane. */     
//         /* If 0 the point is on the plane. If > 0 then the point is on the side of the plane visible along the normal's direction.   */     
//         /* See https://math.stackexchange.com/questions/1330210/how-to-check-if-a-point-is-in-the-direction-of-the-normal-of-a-plane */     
//         vec3 lightMinusFrag = lightPosition - fragPos;                                                                                      
//         float lightRadius = lightData[lightIndex].radius;                                                                                   
//         float distance = length(lightMinusFrag);                                                                                            
                                                                                                                                            
//         if (resamples < MAX_RESAMPLES_PER_PIXEL) {                                                                                          
//             float sideCheck = dot(normal, normalize(lightMinusFrag));                                                                       
//             if (sideCheck < 0.0 || distance > lightRadius) {                                                                                
//                 ++resamples;                                                                                                                
//                 --i;                                                                                                                        
//                 continue;                                                                                                                   
//             }                                                                                                                               
//         }                                                                                                                                   
                                                                                                                                            
//         float distanceRatio = clamp((2.0 * distance) / lightRadius, 0.0, 1.0);                                                              
//         float distAttenuation = distanceRatio;                                                                                              
                                                                                                                                            
//         vec3 lightColor = vec3(50000.0);                                                                                 
                                                                                                                                            
//         float shadowFactor =                                                                                                                
//         distToCamera < 700 ? calculateShadowValue1Sample(probeTextures[entry.index].occlusion,                                                       
//                                                          entry.layer,                                                                       
//                                                          lightData[lightIndex].radius,                                                    
//                                                          fragPos,                                                                           
//                                                          lightPosition,                                                                     
//                                                          dot(lightPosition - fragPos, normal), 0.05)                                        
//                            : 0.0;                                                                                                           
//         shadowFactor = min(shadowFactor, mix(minGiOcclusionFactor, 1.0, distanceRatio));                                                    
                                                                                                                                            
//         float reweightingFactor = 1.0;                                                                                                      
                                                                                                                                            
//         if (shadowFactor > 0.0) {                                                                                                           
//             reweightingFactor = (1.0 - distAttenuation) * minGiOcclusionFactor + distAttenuation;                                           
//         }                                                                                                                                   
                                                                                                                                            
//         validSamples += reweightingFactor;                                                                                                  
                                                                                                                                            
//         vec3 tmpColor = ambientOcclusion * calculateVirtualPointLighting2(fragPos, baseColor, normal, viewDir, specularLightPosition,       
//             lightPosition, lightColor, distToCamera, lightRadius, roughness, metallic, ambient, 0.0, baseReflectivity                       
//         );                                                                                                                                  
//         vplColor = vplColor + (1.0 - shadowFactor) * tmpColor;
//     }

//     validSamples = max(validSamples, 1.0);

//     color = baseColor + PREVENT_DIV_BY_ZERO;//baseColor;
//     reservoir = vec4(boundHDR(vplColor), validSamples);
// }

void main() {
    performLightingCalculations(textureLod(screen, fsTexCoords, 0).rgb, gl_FragCoord.xy, fsTexCoords);
}