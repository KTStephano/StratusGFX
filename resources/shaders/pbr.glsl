STRATUS_GLSL_VERSION

#include "common.glsl"

uniform vec3 infiniteLightDirection;
uniform sampler2DArrayShadow infiniteLightShadowMap;
// Each vec4 offset has two pairs of two (x, y) texel offsets. For each cascade we sample
// a neighborhood of 4 texels and additive blend the results.
uniform vec4 shadowOffset[2];
// Represents a plane which transitions from 0 to 1 as soon as two cascades overlap
uniform vec4 cascadePlanes[3];
uniform mat4 cascadeProjViews[4];

// Example for numSamples: 27, 64, or some other value that can be cleanly cube-rooted
float calculateShadowValue(samplerCube shadowMap, float lightFarPlane, vec3 fragPos, vec3 lightPos, float lightNormalDotProduct, int numSamples) {
    // Not required for fragDir to be normalized
    vec3 fragDir = fragPos - lightPos;
    float currentDepth = length(fragDir);
    // It's very important to multiply by lightFarPlane. The recorded depth
    // is on the range [0, 1] so we need to convert it back to what it was originally
    // or else our depth comparison will fail.
    float calculatedDepth = texture(shadowMap, fragDir).r * lightFarPlane;
    // This bias was found through trial and error... it was originally
    // 0.05 * (1.0 - max...)
    // Part of this came from GPU Gems
    // @see http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch12.html
    float bias = (currentDepth * max(0.5 * (1.0 - max(lightNormalDotProduct, 0.0)), 0.05));// - texture(shadowCubeMap, fragDir).r;
    //float bias = max(0.75 * (1.0 - max(lightNormalDotProduct, 0.0)), 0.05);
    //float bias = max(0.85 * (1.0 - lightNormalDotProduct), 0.05);
    //bias = bias < 0.0 ? bias * -1.0 : bias;
    // Now we use a sampling-based method to look around the current pixel
    // and blend the values for softer shadows (introduces some blur). This falls
    // under the category of Percentage-Closer Filtering (PCF) algorithms.
    float iterationsPerLoop = pow(numSamples, 1.0 / 3.0);
    float shadow = 0.0;
    float totalSamples = 0.0;//PCF_SAMPLES_CUBED; // 64 if samples is set to 4.0
    if (iterationsPerLoop > 2) {
        float offset = 0.2;
        float increment = (2 * offset) / (iterationsPerLoop - 1);
        for (float x = -offset, i = 0; i < iterationsPerLoop; i += 1, x += increment) {
            for (float y = -offset, j = 0; j < iterationsPerLoop; j += 1, y += increment) {
                for (float z = -offset, k = 0; k < iterationsPerLoop; k += 1, z += increment) {
                    float depth = texture(shadowMap, fragDir + vec3(x, y, z)).r;
                    // Perform this operation to go from [0, 1] to
                    // the original value
                    //totalSamples = totalSamples + 1.0;
                    depth = depth * lightFarPlane;
                    if ((currentDepth - bias) > depth) {
                        totalSamples = totalSamples + 1.0;
                        shadow = shadow + 1.0;
                    }
                }
            }
        }
    }

    //float bias = 0.005 * tan(acos(max(lightNormalDotProduct, 0.0)));
    //bias = clamp(bias, 0, 0.01);
    //return (currentDepth - bias) > calculatedDepth ? 1.0 : 0.0;
    return shadow / (totalSamples + 1.0);
}

// See https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch11.html
float sampleShadowTexture(sampler2DArrayShadow shadow, vec4 coords, float depth, vec2 offset, float bias) {
    coords.w = depth - bias;
    coords.xy += offset;
    return texture(shadow, coords);
    // float closestDepth = texture(shadow, coords).r;
    // // 0.0 means not in shadow, 1.0 means fully in shadow
    // return depth > closestDepth ? 1.0 : 0.0;
    // return closestDepth;
}

// For more information, see:
//      "Foundations of Game Development, Volume 2: Rendering", pp. 189
//      https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping
//      https://ogldev.org/www/tutorial49/tutorial49.html
//      https://alextardif.com/shadowmapping.html
//      https://johanmedestrom.wordpress.com/2016/03/18/opengl-cascaded-shadow-maps/
//      https://johanmedestrom.wordpress.com/2016/03/18/opengl-cascaded-shadow-maps/
//      https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch11.html
//      http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-16-shadow-mapping/
float calculateInfiniteShadowValue(vec4 fragPos, vec3 cascadeBlends, vec3 normal) {
	// Since dot(l, n) = cos(theta) when both are normalized, below should compute tan theta
    // See: http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-16-shadow-mapping/
	// float tanTheta = 3.0 * tan(acos(dot(normalize(infiniteLightDirection), normal)));
    // float bias = 0.005 * tanTheta;
    // bias = clamp(bias, 0.0, 0.001);
    float bias = 2e-19;

    vec4 p1, p2;
    vec3 cascadeCoords[4];
    // cascadeCoords[0] = cascadeCoord0 * 0.5 + 0.5;
    for (int i = 0; i < 4; ++i) {
        // cascadeProjViews[i] * fragPos puts the coordinates into clip space which are on the range of [-1, 1].
        // Since we are looking for texture coordinates on the range [0, 1], we first perform the perspective divide
        // and then perform * 0.5 + vec3(0.5).
        vec4 coords = cascadeProjViews[i] * fragPos;
        cascadeCoords[i] = coords.xyz / coords.w; // Perspective divide
        cascadeCoords[i].xyz = cascadeCoords[i].xyz * 0.5 + vec3(0.5);
        // cascadeCoords[i].z = cascadeCoords[i].z * 0.5 + 0.5;
    }

    bool beyondCascade2 = cascadeBlends.y >= 0.0;
    bool beyondCascade3 = cascadeBlends.z >= 0.0;
    // p1.z = float(beyondCascade2) * 2.0;
    // p2.z = float(beyondCascade3) * 2.0 + 1.0;

    int index1 = beyondCascade2 ? 2 : 0;
    int index2 = beyondCascade3 ? 3 : 1;
    p1.z = float(index1);
    p2.z = float(index2);

    vec2 shadowCoord1 = cascadeCoords[index1].xy;
    vec2 shadowCoord2 = cascadeCoords[index2].xy;
    // Convert from range [-1, 1] to [0, 1]
    // shadowCoord1 = shadowCoord1 * 0.5 + 0.5;
    // shadowCoord2 = shadowCoord2 * 0.5 + 0.5;
    float depth1 = cascadeCoords[index1].z;
    float depth2 = cascadeCoords[index2].z;
    // Clamp depths between [0, 1] for final cascade to prevent darkening beyond bounds
    depth2 = beyondCascade3 ? saturate(depth2) : depth2;

    //vec3 blend = saturate(vec3(cascadeBlend[0], cascadeBlend[1], cascadeBlend[2]));
    float weight = beyondCascade2 ? saturate(cascadeBlends.y) - saturate(cascadeBlends.z) : 1.0 - saturate(cascadeBlends.x);

    vec2 wh = computeTexelWidth(infiniteLightShadowMap, 0);
                         
    float light1 = 0.0;
    float light2 = 0.0;
    float samples = 0.0;
    p1.xy = shadowCoord1;
    p2.xy = shadowCoord2;
    // 16-sample filtering - see https://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch11.html
    const float bound = 1.5; // 1.5 = 16 sample; 1.0 = 4 sample
    for (float y = -bound; y <= bound; y += 1.0) {
        for (float x = -bound; x <= bound; x += 1.0) {
            light1 += sampleShadowTexture(infiniteLightShadowMap, p1, depth1, vec2(x, y) * wh, bias);
            light2 += sampleShadowTexture(infiniteLightShadowMap, p2, depth2, vec2(x, y) * wh, bias);
            ++samples;
        }
    }

    // blend and return
    return mix(light2, light1, weight) * (1.0 / samples); //* 0.25;
}