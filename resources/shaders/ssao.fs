#version 410 core

uniform mat4 view;

smooth in vec2 fsTexCoords;

// Structure buffer containing dFdx, dFdy, and z (split into 2 16-bit parts)
uniform sampler2D structureBuffer;
// Allows us to add variation to the pixels we sample to help avoid poor quality
uniform sampler2D rotationLookup;

// GBuffer output
layout (location = 0) out float gLightFactor;

// See https://community.khronos.org/t/saturate/53155
vec3 saturate(vec3 value) {
    return clamp(value, 0.0, 1.0);
}

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

void main() {
    gLightFactor = 0.5;
}