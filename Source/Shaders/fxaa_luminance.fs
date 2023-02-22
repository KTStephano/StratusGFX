// This pass is responsible for per-pixel conversion to a luminance value
// using an equation found here: https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color

// For implementation details see https://catlikecoding.com/unity/tutorials/advanced-rendering/fxaa/

STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "common.glsl"

in vec2 fsTexCoords;

uniform sampler2D screen;

out vec4 color;

// Notice how the green component is favored most heavily since perceptually
// it tends to contribute most to the luminance of a color
float linearColorToLuminance(vec3 linearColor) {
    return 0.2126 * linearColor.r + 0.7152 * linearColor.g + 0.0722 * linearColor.b;
}

void main() {
    vec3 screenColor = texture(screen, fsTexCoords).rgb;
    float luminance = linearColorToLuminance(saturate(screenColor));
    color = vec4(screenColor, luminance);
}