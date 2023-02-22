STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "mesh_data.glsl"

out vec2 fsTexCoords;

void main() {
    fsTexCoords = getTexCoord(gl_VertexID);
    gl_Position = vec4(getPosition(gl_VertexID), 1.0);
}