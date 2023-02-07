STRATUS_GLSL_VERSION

#include "mesh_data.glsl"

//uniform mat4 modelMats[MAX_INSTANCES];
uniform mat4 model;

void main() {
    //mat4 model = modelMats[gl_InstanceID];
    gl_Position = model * vec4(getPosition(gl_VertexID), 1.0);
}