STRATUS_GLSL_VERSION

// Cascaded Shadow Maps
// See https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping

#include "mesh_data.glsl"

uniform mat4 model;
uniform vec3 lightDir;
out float fsTanTheta;

void main () {
	// Since dot(l, n) = cos(theta) when both are normalized, below should compute tan theta
	fsTanTheta = 3.0 * tan(acos(dot(normalize(lightDir), getNormal(gl_VertexID))));
	gl_Position = model * vec4(getPosition(gl_VertexID), 1.0);
}