STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

in vec2 fsTexCoords;

uniform sampler2D screen;

out vec4 color;

void main() {
    color = vec4(texture(screen, fsTexCoords).rgb, 1.0);
}