#version 410 core

uniform sampler2DRect atmosphereBuffer;
// Holds color values after lighting has been applied
uniform sampler2D screenBuffer;
// Contains 2*Lz*(Xlight, Ylight, 1) -- see page 354, eq. 10.81 and page 355, listing 10.14
uniform vec3 lightPosition;

smooth in vec2 fsTexCoords;

out vec4 color;

// This is based on something called "directional median filter." This is the process of taking
// a few samples (3 here) along a certain direction (clip space light direction here) and computing
// the median value of the set of samples taken.
float getAtmosphericIntensity(vec2 pixelCoords) {
    // Remember that lightPosition.z contains 2*Lz
    vec2 direction = normalize(lightPosition.xy - pixelCoords * lightPosition.z);
    float a = texture(atmosphereBuffer, pixelCoords).x;
    float b = texture(atmosphereBuffer, pixelCoords + direction).x;
    float c = texture(atmosphereBuffer, pixelCoords - direction).x;
    // See page 354, eq. 10.83
    return min(max(min(a, b), c), max(a, b));
}

void main() {
    vec2 widthHeight = textureSize(atmosphereBuffer, 0).xy;
    vec3 screenColor = texture(screenBuffer, fsTexCoords).rgb;
    float intensity = getAtmosphericIntensity(fsTexCoords * widthHeight);

    color = vec4(screenColor + intensity * screenColor, 1.0);
}