#version 410 core

smooth in vec2 fsTexCoords;

// Structure buffer containing dFdx, dFdy, and z (split into 2 16-bit parts)
uniform sampler2D structureBuffer;
// Allows us to add variation to the pixels we sample to help avoid poor quality
uniform sampler2D rotationLookup;
uniform float aspectRatio;    // s
uniform float projPlaneZDist; // g
uniform float windowWidth;    // w
uniform float intensity;      // sigma

// GBuffer output
layout (location = 0) out vec3 gLightFactor;

// See https://community.khronos.org/t/saturate/53155
vec3 saturate(vec3 value) {
    return clamp(value, 0.0, 1.0);
}

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

float calculateAmbientOcclusion(vec2 pixelCoords) {
    // In glsl, const refers to constant expression
    const float tau   = 1.0 / 32.0;
    const float dmax  = 2.0;
    float s           = aspectRatio;
    float g           = projPlaneZDist;
    float w           = windowWidth;
    float sigma       = intensity;
    const float r     = 0.4;
    float vectorScale = 2.0 * s / (w * g);

    // Create the base offset vectors
    const float dx[4] = float[](0.25 * r, 0.0    , -0.75 * r, 0.0);
    const float dy[4] = float[](0.0     , 0.5 * r, 0.0      , -r );

    vec4 structure = texture(structureBuffer, pixelCoords);
    float z0 = structure.z + structure.w;

    float scale = z0 * vectorScale;
    vec3 normal = normalize(vec3(structure.xy, -scale));
    scale = 1.0 / scale; // w * g / (2.0 * s)

    // Pull out rotation values so we can calculate final offsets
    vec2 rotation   = texture(rotationLookup, 0.25 * pixelCoords).xy;
    float occlusion = 0.0;
    float weight    = 0.0;

    for (int i = 0; i < 4; ++i) {
        vec3 v;

        // Compute vector v which is the rotated offset vector
        v.x = rotation.x * dx[i] - rotation.y * dy[i]; // delta x camera
        v.y = rotation.y * dx[i] + rotation.x * dy[i]; // delta y camera

        vec2 depth = texture(structureBuffer, pixelCoords + v.xy * scale).zw;
        float z    = depth.x + depth.y;
        v.z        = z - z0;

        // Compute f(v)
        float c  = max(dot(normal, normalize(v)) - tau, 0.0);
        float fv = 1.0 - sqrt(1.0 - c * c);

        // Compute w(v)
        float wv = saturate(1.0 - dot(normal, v) / dmax);

        // Compute H(v)
        float Hv = wv * fv;

        occlusion += Hv;
        weight    += wv;
    }

    // Use 0.0001 for cases when weight is 0 or extremely close to it
    return 1.0 - sigma * occlusion / max(weight, 0.0001);
}

void main() {
    gLightFactor = vec3(calculateAmbientOcclusion(fsTexCoords));
}