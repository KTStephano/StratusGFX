STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
// Enables gl_Layer and gl_ViewportIndex in the vertex shader (no geometry shader required)
#extension GL_ARB_shader_viewport_layer_array : require

#include "mesh_data.glsl"

uniform int layer;

out vec3 fsTexCoords;

uniform mat4 projectionView;

void main() {
    fsTexCoords = getPosition(gl_VertexID);
    vec4 pos = projectionView * vec4(getPosition(gl_VertexID), 1.0);
    // Set the z and w components to w so that w / w = 1.0 which
    // is the maximum depth value. This will allow the graphics pipeline
    // to know that we always want the skybox to fail the depth test if a valid
    // pixel is already in the buffer from a previous stage.
    gl_Position = pos.xyww;

#ifdef USE_LAYERED_RENDERING
    gl_Layer = layer;
#endif
}