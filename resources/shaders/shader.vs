STRATUS_GLSL_VERSION

#include "mesh_data.glsl"

void main() {
    gl_Position = vec4(getPosition(gl_VertexID), 1.0);
}