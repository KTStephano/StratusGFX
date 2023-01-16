STRATUS_GLSL_VERSION

// These needs to match what is in the renderer backend!
// TODO: Find a better way to sync these values with renderer
#define MAX_TOTAL_VPLS_PER_FRAME (128)
#define MAX_VPLS_PER_TILE MAX_TOTAL_VPLS_PER_FRAME

struct VirtualPointLight {
    vec4 lightPosition;
    vec4 lightColor;
    vec4 shadowFarPlaneRadius; // last element padding
    vec4 numShadowSamples; // last 3 elements padding
};