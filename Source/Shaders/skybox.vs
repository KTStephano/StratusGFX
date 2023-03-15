STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "mesh_data.glsl"

out vec3 fsTexCoords;

uniform mat4 projection;
uniform mat4 view;

void main() {
    fsTexCoords = getPosition(gl_VertexID);
    vec4 pos = projection * view * vec4(getPosition(gl_VertexID), 1.0);
    // Set the z and w components to w so that w / w = 1.0 which
    // is the maximum depth value. This will allow the graphics pipeline
    // to know that we always want the skybox to fail the depth test if a valid
    // pixel is already in the buffer from a previous stage.
    gl_Position = pos.xyww;
}