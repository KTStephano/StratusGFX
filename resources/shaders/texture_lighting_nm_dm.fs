#version 150 core

#define MAX_LIGHTS 128
#define SPECULAR_MULTIPLIER 128.0
#define POINT_LIGHT_AMBIENT_INTENSITY 0.05
#define AMBIENT_INTENSITY 0.0005

uniform sampler2D diffuseTexture;
uniform sampler2D normalMap;
uniform sampler2D depthMap;

uniform float shininess = 0.0;
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
 in mat3 fsTbnMatrix;

/**
 * Lighting information. All values related
 * to positions should be in world space.
 */
uniform vec3 lightPositions[MAX_LIGHTS];
uniform vec3 lightColors[MAX_LIGHTS];
// Since max lights is an upper bound, this can
// tell us how many lights are actually present
uniform int numLights = 0;
//uniform float gamma = 2.2;

out vec4 fsColor;

vec3 calculatePointLighting(vec3 baseColor, vec3 normal, vec3 viewDir, int lightIndex) {
    vec3 lightPos = lightPositions[lightIndex];
    vec3 lightColor = lightColors[lightIndex];

    vec3 lightDir = lightPos - fsPosition;
    float lightDist = length(lightDir);
    lightDir = normalize(lightDir);
    // Linear attenuation
    float attenuationFactor = 1 / (lightDist * lightDist);

    float lightNormalDot = max(dot(lightDir, normal), 0.0);
    vec3 ambient = POINT_LIGHT_AMBIENT_INTENSITY * lightColor * baseColor;
    vec3 diffuse = lightNormalDot * lightColor * baseColor;

    vec3 halfAngleDir = normalize(lightDir + viewDir);
    float exponent = max(shininess * SPECULAR_MULTIPLIER, 8);
    vec3 specular = pow(
        max(dot(normal, halfAngleDir), 0.0),
        exponent) * lightColor * baseColor;
    
    return (ambient + diffuse + specular) * attenuationFactor;
}

// See https://learnopengl.com/Advanced-Lighting/Parallax-Mapping
vec2 calculateDepthCoords(vec2 texCoords, vec3 viewDir) {
    float height = texture(depthMap, texCoords).r;
    vec2 p = viewDir.xy * (height * heightScale);
    return texCoords - p;
}

void main() {
    vec3 viewDir = normalize(viewPosition - fsPosition);
    vec2 texCoords = fsTexCoords;
    texCoords = calculateDepthCoords(texCoords, viewDir);
    //if(texCoords.x > 1.0 || texCoords.y > 1.0 || texCoords.x < 0.0 || texCoords.y < 0.0) {
    //    discard;
    //}

    vec3 baseColor = texture(diffuseTexture, texCoords).rgb;
    vec3 normal = texture(normalMap, texCoords).rgb;
    // Normals generally have values from [-1, 1], but inside
    // an OpenGL texture they are transformed to [0, 1]. To convert
    // them back, we multiply by 2 and subtract 1.
    normal = normalize(normal * 2.0 - 1.0);
    normal = normalize(fsTbnMatrix * normal);
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