STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "mesh_data.glsl"

void main() {
    gl_Position = vec4(getPosition(gl_VertexID), 1.0);
}