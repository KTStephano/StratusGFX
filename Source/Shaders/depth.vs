STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "mesh_data.glsl"

uniform mat4 projectionView;

void main() {
    gl_Position = projectionView * model * vec4(getPosition(gl_VertexID), 1.0);
}