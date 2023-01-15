STRATUS_GLSL_VERSION

// These needs to match what is in the renderer backend!
// TODO: Find a better way to sync these values with renderer
#define MAX_VPLS_PER_TILE (4)
#define MAX_TOTAL_VPLS_PER_FRAME (300)

struct VirtualPointLight {
    vec4 lightPosition;
    vec4 lightColor;
    float shadowFactor;
    float lightFarPlane;
    float lightRadius;
    float _1;
    vec4 _padding;
};