STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "mesh_data.glsl"
#include "common.glsl"

layout (std430, binding = CURR_FRAME_MODEL_MATRICES_BINDING_POINT) readonly buffer SSBO3 {
    mat4 modelMatrices[];
};

uniform mat4 projectionView;

smooth out vec2 fsTexCoords;

flat out int fsDiffuseMapped;
flat out int fsDrawID;

void main() {
    Material material = materials[materialIndices[gl_DrawID]];
    uint flags = material.flags;

    fsDiffuseMapped = int(bitwiseAndBool(flags, GPU_DIFFUSE_MAPPED));
    fsTexCoords = getTexCoord(gl_VertexID);
    fsDrawID = gl_DrawID;

    gl_Position = projectionView * modelMatrices[gl_DrawID] * vec4(getPosition(gl_VertexID), 1.0);
}