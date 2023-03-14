STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

uniform mat4 projection;
uniform mat4 view;

void main() {
	gl_Position = projection * view * vec4(1.0);
}