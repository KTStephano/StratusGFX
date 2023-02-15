STRATUS_GLSL_VERSION

#include "mesh_data.glsl"

//uniform mat4 modelMats[MAX_INSTANCES];

layout (std430, binding = 13) readonly buffer SSBO3 {
    mat4 modelMatrices[];
};

void main() {
    //mat4 model = modelMats[gl_InstanceID];
    gl_Position = modelMatrices[gl_DrawID] * vec4(getPosition(gl_VertexID), 1.0);
}