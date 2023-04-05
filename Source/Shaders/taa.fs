// See this page: https://sugulee.wordpress.com/2021/06/21/temporal-anti-aliasingtaa-tutorial/
// https://ziyadbarakat.wordpress.com/2020/07/28/temporal-anti-aliasing-step-by-step/
// https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail/
// http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/

STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "common.glsl"
#include "aa_common.glsl"

in vec2 fsTexCoords;

uniform sampler2D screen;
uniform sampler2D prevScreen;
uniform sampler2D velocity;

out vec4 color;

#define MAX_COLOR_DIFFERENCE 0.1

void main() {
    vec2 velocityVal = texture(velocity, fsTexCoords).xy;
    // Convert from [0, 1] to [-1, 1]
    //velocityVal = velocityVal * 2.0 - 1.0;
    // Adjust based on texture size
    //velocityVal /= vec2(textureSize(screen, 0).xy);
    vec2 prevTexCoords = fsTexCoords - velocityVal;

    vec3 currentColor = texture(screen, fsTexCoords).rgb;
    vec3 prevColor = texture(prevScreen, prevTexCoords).rgb;

    // Collect information around the texture coordinate and use it
    // to apply clamping (otherwise we get extreme ghosting)
    vec3 prevColor0 = textureOffset(screen, fsTexCoords, ivec2( 0,  1)).rgb;
    vec3 prevColor1 = textureOffset(screen, fsTexCoords, ivec2( 0, -1)).rgb;
    vec3 prevColor2 = textureOffset(screen, fsTexCoords, ivec2( 1,  0)).rgb;
    vec3 prevColor3 = textureOffset(screen, fsTexCoords, ivec2(-1,  0)).rgb;

    vec3 minColor = min(currentColor, min(prevColor0, min(prevColor1, min(prevColor2, prevColor3))));
    vec3 maxColor = max(currentColor, max(prevColor0, max(prevColor1, max(prevColor2, prevColor3))));

    prevColor = clamp(prevColor, minColor, maxColor);

    // vec3 minColor = vec3(FLOAT_MAX);
    // vec3 maxColor = vec3(-FLOAT_MAX);
 
    // // Sample a 3x3 neighborhood to create a box in color space
    // for(int x = -1; x <= 1; ++x)
    // {
    //     for(int y = -1; y <= 1; ++y)
    //     {
    //         vec3 color = textureOffset(screen, fsTexCoords, ivec2(x, y)).rgb;
    //         minColor = min(minColor, color);
    //         maxColor = max(maxColor, color);
    //     }
    // }

    // prevColor = clamp(prevColor, minColor, maxColor);

    //color = vec4(currentColor, 1.0);
    color = vec4(mix(currentColor, prevColor, 0.9), 1.0);
}