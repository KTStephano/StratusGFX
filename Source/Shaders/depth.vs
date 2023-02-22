STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "mesh_data.glsl"

uniform mat4 projection;
uniform mat4 view;

void main() {
    gl_Position = projection * view * model * vec4(getPosition(gl_VertexID), 1.0);
}