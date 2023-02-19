STRATUS_GLSL_VERSION

in vec3 fsTexCoords;

layout (location = 0) out vec4 fsColor;

uniform samplerCube skybox;

void main() {
    fsColor = texture(skybox, fsTexCoords);
}