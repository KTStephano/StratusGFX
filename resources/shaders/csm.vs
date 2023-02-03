STRATUS_GLSL_VERSION

// Cascaded Shadow Maps
// See https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping

layout (location = 0)  in vec3 position;
layout (location = 2)  in vec3 normal;
//layout (location = 12) in mat4 model;

uniform mat4 model;
uniform vec3 lightDir;
out float fsTanTheta;

void main () {
	// Since dot(l, n) = cos(theta) when both are normalized, below should compute tan theta
	fsTanTheta = 3.0 * tan(acos(dot(normalize(lightDir), normal)));
	gl_Position = model * vec4(position, 1.0);
}