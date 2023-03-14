STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "aabb.glsl"
#include "mesh_data.glsl"

layout (std430, binding = 15) readonly buffer SSBO2 {
    mat4 globalMatrices[];
};

layout (std430, binding = 13) readonly buffer SSBO3 {
    mat4 modelMatrices[];
};

layout (std430, binding = 14) readonly buffer inputBlock3 {
    AABB aabbs[];
};

uniform mat4 projection;
uniform mat4 view;
uniform int modelIndex;

void main() {
    AABB aabb = transformAabb(aabbs[modelIndex], modelMatrices[modelIndex]);
    vec4 corners[8] = computeCornersWithTransform(aabb, projection * view);
    vec4 vertices[24] = convertCornersToLineVertices(corners);

    gl_Position = vertices[gl_VertexID];
}