#version 150 core

#define MAX_LIGHTS 128
#define SPECULAR_MULTIPLIER 64.0
#define AMBIENT_INTENSITY 0.005

uniform sampler2D diffuseTexture;
uniform float shininess = 0.0;

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
 * Lighting information. All values related
 * to positions should be in world space.
 */
uniform vec3 lightPositions[MAX_LIGHTS];
uniform vec3 lightColors[MAX_LIGHTS];
// Since max lights is an upper bound, this can
// tell us how many lights are actually present
uniform int numLights = 0;
uniform float gamma = 2.2;

out vec4 fsColor;

vec3 calculatePointLighting(vec3 baseColor, vec3 viewDir, int lightIndex) {
    vec3 lightPos = lightPositions[lightIndex];
    vec3 lightColor = lightColors[lightIndex];

    vec3 lightDir = lightPos - fsPosition;
    float lightDist = length(lightDir);
    lightDir = normalize(lightDir);
    // Linear attenuation
    float attenuationFactor = 1 / (lightDist);

    float lightNormalDot = dot(lightDir, fsNormal);
    vec3 ambient = AMBIENT_INTENSITY * lightColor * baseColor;
    vec3 diffuse = max(lightNormalDot, 0.0) * lightColor * baseColor;

    vec3 halfAngleDir = normalize(lightDir + viewDir);
    vec3 specular = pow(
        max(dot(fsNormal, halfAngleDir), 0.0),
        shininess * SPECULAR_MULTIPLIER) * lightColor * baseColor;
    
    return (ambient + diffuse + specular) * attenuationFactor;
}

void main() {
    vec3 baseColor = texture(diffuseTexture, fsTexCoords).rgb;
    vec3 viewDir = normalize(viewPosition - fsPosition);
    vec3 color = calculatePointLighting(baseColor, viewDir, 0);
    color = color + baseColor * AMBIENT_INTENSITY;
    // Apply gamma correction
    color = pow(color, vec3(1.0 / gamma));
    fsColor = vec4(color, 1.0);
}