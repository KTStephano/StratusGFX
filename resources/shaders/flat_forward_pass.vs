STRATUS_GLSL_VERSION

#include "mesh_data.glsl"

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;
//uniform mat4 modelView;

smooth out vec2 fsTexCoords;

void main() {
    gl_Position = projection * view * model * vec4(getPosition(gl_VertexID), 1.0);
    fsTexCoords = getTexCoord(gl_VertexID);
}