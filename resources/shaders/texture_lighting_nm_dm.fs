#version 410 core

// Apple limits us to 16 total samplers active in the pipeline :(
#define MAX_LIGHTS 12
#define SPECULAR_MULTIPLIER 128.0
#define POINT_LIGHT_AMBIENT_INTENSITY 0.05
#define AMBIENT_INTENSITY 0.0005

uniform sampler2D diffuseTexture;
uniform sampler2D normalMap;
uniform sampler2D depthMap;

//uniform float fsShininessVals[MAX_INSTANCES];
//uniform float fsShininess = 0.0;
in float fsShininess;
uniform float heightScale = 0.1;

/**
 * Information about the camera
 */
uniform vec3 viewPosition;

/**
 * Fragment information. All values should be
 * in world space.
 */
in vec3 fsPosition;
in vec3 fsNormal;
in vec2 fsTexCoords;
in mat4 fsModel;
//in float fsfsShininess;

/**
 * Tangent space
 */
in mat3 fsTbnMatrix;
in vec3 fsTanViewPosition;
in vec3 fsTanFragPosition;

/**
 * Lighting information. All values related
 * to positions should be in world space.
 */
uniform vec3 lightPositions[MAX_LIGHTS];
uniform vec3 lightColors[MAX_LIGHTS];
uniform samplerCube shadowCubeMaps[MAX_LIGHTS];
uniform float lightFarPlanes[MAX_LIGHTS];
// Since max lights is an upper bound, this can
// tell us how many lights are actually present
uniform int numLights = 0;
//uniform float gamma = 2.2;

out vec4 fsColor;

float calculateShadowValue(vec3 fragPos, vec3 lightPos, int lightIndex, float lightNormalDotProduct) {
    // Not required for fragDir to be normalized
    vec3 fragDir = fragPos - lightPos;
    float currentDepth = length(fragDir);
    // It's very important to multiply by lightFarPlane. The recorded depth
    // is on the range [0, 1] so we need to convert it back to what it was originally
    // or else our depth comparison will fail.
    float calculatedDepth = texture(shadowCubeMaps[lightIndex], fragDir).r * lightFarPlanes[lightIndex];
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
    float shadow = 0.0;
    float samples = 4.0;
    float totalSamples = samples * samples * samples; // 64 if samples is set to 4.0
    float offset = 0.1;
    float increment = offset / (samples * 0.5);
    for (float x = -offset; x < offset; x += increment) {
        for (float y = -offset; y < offset; y += increment) {
            for (float z = -offset; z < offset; z += increment) {
                float depth = texture(shadowCubeMaps[lightIndex], fragDir + vec3(x, y, z)).r;
                // Perform this operation to go from [0, 1] to
                // the original value
                depth = depth * lightFarPlanes[lightIndex];
                if ((currentDepth - bias) > depth) {
                    shadow = shadow + 1.0;
                }
            }
        }
    }

    //float bias = 0.005 * tan(acos(max(lightNormalDotProduct, 0.0)));
    //bias = clamp(bias, 0, 0.01);
    //return (currentDepth - bias) > calculatedDepth ? 1.0 : 0.0;
    return shadow / totalSamples;
}

vec3 calculatePointLighting(vec3 baseColor, vec3 normal, vec3 viewDir, int lightIndex) {
    vec3 lightPos = fsTbnMatrix * lightPositions[lightIndex];
    vec3 lightColor = lightColors[lightIndex];

    vec3 lightDir = lightPos - fsTanFragPosition;
    float lightDist = length(lightDir);
    lightDir = normalize(lightDir);
    // Linear attenuation
    float attenuationFactor = 1 / (lightDist * lightDist);

    float lightNormalDot = max(dot(lightDir, normal), 0.0);
    vec3 ambient = POINT_LIGHT_AMBIENT_INTENSITY * lightColor * baseColor;
    vec3 diffuse = lightNormalDot * lightColor * baseColor;

    vec3 halfAngleDir = normalize(lightDir + viewDir);
    float exponent = max(fsShininess * SPECULAR_MULTIPLIER, 8);
    vec3 specular = pow(
        max(dot(normal, halfAngleDir), 0.0),
        exponent) * lightColor * baseColor;
    // We need to perform shadow calculations in world space
    float shadowFactor = calculateShadowValue(fsPosition, lightPositions[lightIndex],
        lightIndex, dot(lightPositions[lightIndex] - fsPosition, fsNormal));
    //float shadowFactor = calculateShadowValue(fsTanFragPosition, lightPos,
    //   lightIndex, lightNormalDot);
    //float shadowFactor = calculateShadowValue(fsPosition, lightIndex);

    return (ambient + (1.0 - shadowFactor) * (diffuse + specular)) * attenuationFactor;
}

// See https://learnopengl.com/Advanced-Lighting/Parallax-Mapping
vec2 calculateDepthCoords(vec2 texCoords, vec3 viewDir) {
    float height = texture(depthMap, texCoords).r;
    vec2 p = viewDir.xy * (height * 0.05);
    return texCoords - p;
}

void main() {
    vec3 viewDir = normalize(fsTanViewPosition - fsTanFragPosition);
    vec2 texCoords = fsTexCoords;
    texCoords = calculateDepthCoords(texCoords, viewDir);
    if(texCoords.x > 1.0 || texCoords.y > 1.0 || texCoords.x < 0.0 || texCoords.y < 0.0) {
       discard;
    }

    vec3 baseColor = texture(diffuseTexture, texCoords).rgb;
    vec3 normal = mat3(fsModel) * texture(normalMap, texCoords).rgb;
    // Normals generally have values from [-1, 1], but inside
    // an OpenGL texture they are transformed to [0, 1]. To convert
    // them back, we multiply by 2 and subtract 1.
    normal = normalize(normal * 2.0 - 1.0);
    //vec3 tbnNormal = normalize(fsTbnMatrix * normal);
    //normal = normalize(fsTbnMatrix * normal);
    vec3 color = vec3(0.0);
    for (int i = 0; i < MAX_LIGHTS; ++i) {
        if (i >= numLights) break;
        color = color + calculatePointLighting(baseColor, normal, viewDir, i);
    }
    color = color + baseColor * AMBIENT_INTENSITY;
    //vec3 color = calculatePointLighting(baseColor, normal, viewDir, 0);
    //color = color + baseColor * AMBIENT_INTENSITY;
    // Apply gamma correction
    //color = pow(color, vec3(1.0 / gamma));
    fsColor = vec4(color, 1.0);
}