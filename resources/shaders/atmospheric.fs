#version 410 core

#define NUM_CASCADES 4

// Structure buffer containing dFdx, dFdy, and z (split into 2 16-bit parts)
uniform sampler2DRect structureBuffer;
uniform sampler2DArrayShadow infiniteLightShadowMap;
uniform float maxCascadeDepth[4];
uniform mat4 cascade0ToCascadeK[NUM_CASCADES - 1];
// This should be something like 32x32 or 64x64
uniform sampler2D noiseTexture;
uniform float minAtmosphereDepth;            // dmin
uniform float atmosphereDepthDiff;           // dmax - dmin
uniform float atmosphereDepthRatio;          // dmax / dmin
uniform float atmosphereFogDensity;          // lambda
uniform vec3 anisotropyConstants;            // 1 - g, 1 + g * g, 2 * g
uniform vec4 shadowSpaceCameraPos;           // M_shadow(0) * M_camera[3]
uniform vec3 normalizedCameraLightDirection; // This is in camera space and is the direction from the camera to the light
uniform vec2 noiseShift;                     // Should be randomized each frame on the range [0, 1 / numSamples]
uniform float numSamples = 64.0;             // Number of samples to take along each ray we cast
uniform float windowWidth;
uniform float windowHeight;

smooth in vec2 fsTexCoords;
// fsCamSpaceRay is always used as vec3(fsCamSpaceRay.xy, minAtmosphereDepth)
smooth in vec2 fsCamSpaceRay;
smooth in vec3 fsShadowSpaceRay;

// GBuffer output
layout (location = 0) out vec3 gAtmosphere;

// Linear interpolate
vec4 lerp(vec4 x, vec4 y, float a) {
    return mix(x, y, a);
}

vec3 lerp(vec3 x, vec3 y, float a) {
    return mix(x, y, a);
}

float lerp(float x, float y, float a) {
    return mix(x, y, a);
}

// Calculates p1, p1 from page 342, eq. 10.64
// invLength = 1.0 / length(camSpaceRayDir)
// scale = minAtmosphereDepth * invLength
void calculateShadowSpaceMinMaxRayPoints(vec2 camSpaceRayDir, vec3 shadowSpaceRayDir, out float invLength, out float scale, out vec4 p1, out vec4 p2) {
    invLength = 1.0 / length(vec3(camSpaceRayDir, minAtmosphereDepth));
    scale = minAtmosphereDepth * invLength;

    p1 = vec4(scale * shadowSpaceRayDir, 0.0);
    p2 = shadowSpaceCameraPos + atmosphereDepthRatio * p1;
    p1 += shadowSpaceCameraPos;
}

// Calculates z1, z2 from page 343, eq. 10.65
// invLength = 1.0 / length(camSpaceRayDir)
// scale = minAtmosphereDepth * invLength
// z1 = minAtmosphereDepth * scale
// z2 = atmosphereDepthRatio * z1
void calculateCameraSpaceMinMaxDepths(float invLength, float scale, out float z1, out float z2) {
    z1 = minAtmosphereDepth * scale;
    z2 = atmosphereDepthRatio * z1;
}

// Uses the normalized Henyey-Greenstein phase function from page 348, eq. 10.79
void calculateNormalizedAnisotropicScatteringIntensity(vec2 camSpaceRayDir, float invLengthCamSpaceRayDir, out float atmosphereBrightness, out float anisotropicScattering, out float intensity) {
    // Page 345, eq. 10.73
    atmosphereBrightness = atmosphereFogDensity * atmosphereDepthDiff / (numSamples + 1);
    // Page 348, eq. 10.77
    float cosA = dot(normalizedCameraLightDirection, vec3(camSpaceRayDir, minAtmosphereDepth)) * invLengthCamSpaceRayDir;
    float nhg = anisotropyConstants.x * inversesqrt(anisotropyConstants.y - anisotropyConstants.z * cosA);
    
    anisotropicScattering = nhg * nhg * nhg;
    intensity = atmosphereBrightness * anisotropicScattering;
}

// u(t) from page 343, eq. 10.67
float UofT(float t, float m) {
    return t * ((1.0 - m) * t + m);
}

// See https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch11.html
float sampleShadowTexture(sampler2DArrayShadow shadow, vec4 coords, float cascadeSwitch, float bias) {
    coords.xyz = coords.xyz / coords.w; // Perspective divide
    coords.xyz = coords.xyz * 0.5 + vec3(0.5); // Convert to range [0, 1]
    
    // Set depth to be the w coordinate
    coords.w = coords.z - bias;
    coords.z = cascadeSwitch;

    return texture(shadow, coords);
}

// Calculates the final brightness of this pixel which is Brightness * Normalized Anisotropic Scattering Intensity,
// which is a combination of page 345, eq. 10.74 and page 348, eq. 10.79
float calculateFinalBrightness(vec2 pixelCoords, float z1, float z2, vec4 p1, vec4 p2, float anisotropicScatteringIntensity) {
    const float m = 0.25; // slope (first sample)
    float deltaT  = 1.0 / numSamples;
    // Page 345
    float deltaW  = 2 * (1 - m) * deltaT;

    // Combine zw into approximate depth from structure buffer
    // Note that here we are assuming that the atmosphere buffer is the same dimensions as the structure buffer!
    // If they are less (ex: half size like in the book) then the pixelCoords will need to be multiplied by some constant
    vec2 structure = texture(structureBuffer, pixelCoords).zw;
    float depth = structure.x + structure.y;

    float inverseNoiseDim = 1.0 / textureSize(noiseTexture, 0).x;
    // Page 346, eq. 10.75
    float t = deltaT * texture(noiseTexture, pixelCoords * inverseNoiseDim + noiseShift).x;
    float tmax = t + 1.0;
    float atmosphere = 0.0; // Accumulates atmosphere value
    float weight = m; // First weight always starts with m

    // We start walking the ray with cascade 0
    float cascadeDepthSwitch = 0.0;
    float zmax = min(depth, maxCascadeDepth[0]);
    float bias = 2e-19;
    for (; t <= tmax; t += deltaT) {
        float u = UofT(t, m);
        float z = lerp(z1, z2, u);
        // Never exceed max depth in structure buffer
        if (z >= zmax) {
            break;
        }
                                                        
        // Calculate sampling location - remember p1 and p2 are in cascade 0 space
        vec4 shadowCoords = lerp(p1, p2, u);
        atmosphere += weight * sampleShadowTexture(infiniteLightShadowMap, shadowCoords, cascadeDepthSwitch, bias);
        weight += deltaW;
    }

    for (int cascade = 1; cascade < NUM_CASCADES; ++cascade) {
        // This is the switch determining which texture segment of the shadow map we sample from
        cascadeDepthSwitch = float(cascade);
        zmax = min(depth, maxCascadeDepth[cascade]);

        for (; t <= zmax; t += deltaT) {
            float u = UofT(t, m);
            float z = lerp(z1, z2, u);
            // Never exceed max depth in structure buffer
            if (z >= zmax) {
                break;
            }

            // Calculate sampling location - remember p1 and p2 are in cascade 0 space so we need to transform
            // them to cascade current space
            vec4 shadowCoords = cascade0ToCascadeK[cascade - 1] * lerp(p1, p2, u);
            atmosphere += weight * sampleShadowTexture(infiniteLightShadowMap, shadowCoords, cascadeDepthSwitch, bias);
            weight += deltaW;
        }
    }

    return anisotropicScatteringIntensity * atmosphere;
}

void main() {
    float invLength;
    float scale;
    vec4 p1;
    vec4 p2;
    float z1;
    float z2;

    calculateShadowSpaceMinMaxRayPoints(fsCamSpaceRay, fsShadowSpaceRay, invLength, scale, p1, p2);
    calculateCameraSpaceMinMaxDepths(invLength, scale, z1, z2);

    float atmosphereBrightness;
    float atmosphereScattering;
    float intensity;
    calculateNormalizedAnisotropicScatteringIntensity(fsCamSpaceRay, invLength, atmosphereBrightness, atmosphereScattering, intensity);

    float finalBrightness = calculateFinalBrightness(fsTexCoords * vec2(windowWidth, windowHeight), z1, z2, p1, p2, intensity);

    gAtmosphere = vec3(finalBrightness);
    //gAtmosphere = vec3(atmosphereBrightness, atmosphereScattering, intensity);
}