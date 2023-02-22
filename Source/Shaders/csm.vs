STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

// Cascaded Shadow Maps
// See Foundations of Game Engine Development Volume 2 (section on cascaded shadow maps)
// See https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping

// Enables gl_Layer and gl_ViewportIndex in the vertex shader (no geometry shader required)
#extension GL_ARB_shader_viewport_layer_array : require

#include "mesh_data.glsl"

layout (std430, binding = 13) readonly buffer SSBO3 {
    mat4 modelMatrices[];
};

uniform mat4 shadowMatrix;

uniform vec3 lightDir;
uniform int depthLayer;
out float fsTanTheta;

void main () {
	// Select which layer of the depth texture we will write to
	// (DEPTH_LAYER is defined in C++ code)
	gl_Layer = DEPTH_LAYER;

	// Since dot(l, n) = cos(theta) when both are normalized, below should compute tan theta
	fsTanTheta = 3.0 * tan(acos(dot(normalize(lightDir), getNormal(gl_VertexID))));
	gl_Position = shadowMatrix * modelMatrices[gl_DrawID] * vec4(getPosition(gl_VertexID), 1.0);
}