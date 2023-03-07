STRATUS_GLSL_VERSION

// Axis-Aligned Bounding Box
struct AABB {
    vec4 vmin;
    vec4 vmax;
    float center;
    float size;
    float _1[2];
};