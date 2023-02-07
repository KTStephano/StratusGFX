STRATUS_GLSL_VERSION

#include "mesh_data.glsl"

smooth out vec2 fsTexCoords;

void main() {
    fsTexCoords = getTexCoord(gl_VertexID);
    gl_Position = vec4(getPosition(gl_VertexID), 1.0);
}