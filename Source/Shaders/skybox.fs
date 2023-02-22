STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

in vec3 fsTexCoords;

layout (location = 0) out vec4 fsColor;

uniform samplerCube skybox;

void main() {
    fsColor = texture(skybox, fsTexCoords);
}