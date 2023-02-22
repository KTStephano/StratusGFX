STRATUS_GLSL_VERSION

// Enables gl_Layer and gl_ViewportIndex in the vertex shader (no geometry shader required)
#extension GL_ARB_shader_viewport_layer_array : require

#include "mesh_data.glsl"

uniform mat4 shadowMatrix;

layout (std430, binding = 13) readonly buffer SSBO3 {
    mat4 modelMatrices[];
};

smooth out vec4 fsPosition;

void main() {
    // Select which layer of the depth texture we will write to
	// (DEPTH_LAYER is defined in C++ code)
	gl_Layer = DEPTH_LAYER;

    fsPosition = modelMatrices[gl_DrawID] * vec4(getPosition(gl_VertexID), 1.0);
    gl_Position = shadowMatrix * fsPosition;
}