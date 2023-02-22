STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "mesh_data.glsl"

layout (std430, binding = 13) readonly buffer SSBO3 {
    mat4 modelMatrices[];
};

uniform mat4 projection;
uniform mat4 view;
//uniform mat4 modelView;

smooth out vec2 fsTexCoords;
flat out int fsDrawID;

void main() {
    gl_Position = projection * view * modelMatrices[gl_DrawID] * vec4(getPosition(gl_VertexID), 1.0);
    fsTexCoords = getTexCoord(gl_VertexID);
    fsDrawID = gl_DrawID;
}