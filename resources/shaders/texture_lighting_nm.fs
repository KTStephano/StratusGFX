#version 410 core

#define MAX_LIGHTS 128
#define SPECULAR_MULTIPLIER 128.0
#define POINT_LIGHT_AMBIENT_INTENSITY 0.05
#define AMBIENT_INTENSITY 0.0005

uniform sampler2D diffuseTexture;
uniform sampler2D normalMap;
//uniform float fsShininess = 0.0;
in float fsShininess;

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
// Since max lights is an upper bound, this can
// tell us how many lights are actually present
uniform int numLights = 0;
//uniform float gamma = 2.2;

out vec4 fsColor;

vec3 calculatePointLighting(vec3 baseColor, vec3 normal, vec3 viewDir, int lightIndex) {
    vec3 lightPos = lightPositions[lightIndex];
    vec3 lightColor = lightColors[lightIndex];

    vec3 lightDir = (fsTbnMatrix * lightPos) - fsTanFragPosition;
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
    
    return (ambient + diffuse + specular) * attenuationFactor;
}

void main() {
    vec3 baseColor = texture(diffuseTexture, fsTexCoords).rgb;
    vec3 viewDir = normalize(fsTanViewPosition - fsTanFragPosition);
    vec3 normal = texture(normalMap, fsTexCoords).rgb;
    // Normals generally have values from [-1, 1], but inside
    // an OpenGL texture they are transformed to [0, 1]. To convert
    // them back, we multiply by 2 and subtract 1.
    normal = normalize(normal * 2.0 - 1.0);
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