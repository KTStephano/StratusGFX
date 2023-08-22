STRATUS_GLSL_VERSION

#include "pbr.glsl"

// These needs to match what is in the renderer backend!
// TODO: Find a better way to sync these values with renderer
#define MAX_TOTAL_VPLS_BEFORE_CULLING (10000)
#define MAX_TOTAL_VPLS_PER_FRAME (MAX_TOTAL_SHADOW_MAPS)
#define MAX_VPLS_PER_TILE (12)

// Needs to match up with definition in StratusGpuCommon
struct VplStage1PerTileOutputs {
    vec4 averageLocalPosition;
    vec4 averageLocalNormal;
};

struct VplStage2PerTileOutputs {
    int numVisible;
    int indices[MAX_VPLS_PER_TILE];
};

struct VplData {
    vec4 position;
    vec4 color;
    vec4 specularPosition;
    float radius;
    float farPlane;
    float intensity;
    float _1;
};