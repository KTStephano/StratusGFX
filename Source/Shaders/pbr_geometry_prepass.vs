STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "mesh_data.glsl"
#include "common.glsl"

layout (std430, binding = 13) readonly buffer SSBO3 {
    mat4 modelMatrices[];
};

layout (std430, binding = 14) readonly buffer SSBO4 {
    mat4 prevModelMatrices[];
};

uniform mat4 jitterProjectionView;

smooth out vec2 fsTexCoords;
flat out int fsDiffuseMapped;
flat out int fsDrawID;

void main() {
    Material material = materials[materialIndices[gl_DrawID]];
    uint flags = material.flags;

    fsDiffuseMapped = int(bitwiseAndBool(flags, GPU_DIFFUSE_MAPPED));
    fsTexCoords = getTexCoord(gl_VertexID);
    fsDrawID = gl_DrawID;

    gl_Position = jitterProjectionView * modelMatrices[gl_DrawID] * vec4(getPosition(gl_VertexID), 1.0);
}