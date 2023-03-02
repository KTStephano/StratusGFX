STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

in vec3 fsTexCoords;

layout (location = 0) out vec4 fsColor;

uniform samplerCube skybox;
uniform vec3 colorMask = vec3(1.0);
uniform float intensity = 3.0;

void main() {
    fsColor = intensity * vec4(colorMask, 1.0) * texture(skybox, fsTexCoords);
}