// See this page: https://sugulee.wordpress.com/2021/06/21/temporal-anti-aliasingtaa-tutorial/
// https://ziyadbarakat.wordpress.com/2020/07/28/temporal-anti-aliasing-step-by-step/
// https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail/
// http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
// https://de45xmedrsdbp.cloudfront.net/Resources/files/TemporalAA_small-59732822.pdf

STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "common.glsl"
#include "aa_common.glsl"

in vec2 fsTexCoords;

uniform sampler2D screen;
uniform sampler2D prevScreen;
uniform sampler2D velocity;
uniform sampler2D previousVelocity;

out vec4 color;

#define MAX_COLOR_DIFFERENCE 0.1

void main() {
    vec2 velocityVal = texture(velocity, fsTexCoords).xy;
    // Convert from [0, 1] to [-1, 1]
    //velocityVal = velocityVal * 2.0 - 1.0;
    // Adjust based on texture size
    //velocityVal /= vec2(textureSize(screen, 0).xy);
    vec2 prevTexCoords = fsTexCoords - velocityVal;
    // vec2 prevVelocityVal = texture(previousVelocity, prevTexCoords).xy;

    // float velocityDifference = length(prevVelocityVal - velocityVal);

    vec3 currentColor = texture(screen, fsTexCoords).rgb;
    vec3 prevColor = tonemap(texture(prevScreen, prevTexCoords).rgb);

    // Collect information around the texture coordinate and use it 
    // to apply clamping (otherwise we get extreme ghosting)
    vec3 currColor1 = textureOffset(screen, fsTexCoords, ivec2( 0,  1)).rgb;
    vec3 currColor2 = textureOffset(screen, fsTexCoords, ivec2( 0, -1)).rgb;
    vec3 currColor3 = textureOffset(screen, fsTexCoords, ivec2( 1,  0)).rgb;
    vec3 currColor4 = textureOffset(screen, fsTexCoords, ivec2(-1,  0)).rgb;

    vec3 minColor = tonemap(min(currentColor, min(currColor1, min(currColor2, min(currColor3, currColor4)))));
    vec3 maxColor = tonemap(max(currentColor, max(currColor1, max(currColor2, max(currColor3, currColor4)))));

    // vec3 minColor = currentColor;
    // vec3 maxColor = currentColor;
    // for(int x = -1; x <= 1; ++x)
    // {
    //     for(int y = -1; y <= 1; ++y)
    //     {
    //         vec3 color = textureOffset(screen, fsTexCoords, ivec2(x, y)).rgb;
    //         minColor = min(minColor, color);
    //         maxColor = max(maxColor, color);
    //     }
    // }

    // minColor = tonemap(minColor);
    // maxColor = tonemap(maxColor);

    // float minLuminance = linearColorToLuminance(minColor);
    // float maxLuminance = linearColorToLuminance(maxColor);

    // float prevLuminance = linearColorToLuminance(prevColor);

    // float luminance = clamp(prevLuminance, minLuminance, maxLuminance);
    // float difference = abs(luminance - prevLuminance);
    // float weight = difference > 0.05 ? 0.1 : 0.9;

    currentColor = tonemap(currentColor);

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
    // float velocityDisocclusion = saturate((velocityDifference - 0.001) * 10.0);
    //vec3 averageCurrentColor = (currentColor + currColor1 + currColor2 + currColor3 + currColor4) / 5.0;

    currentColor = inverseTonemap(currentColor);
    prevColor = inverseTonemap(prevColor);

    //color = vec4(currentColor, 1.0);
    //color = vec4(averageCurrentColor, 1.0);
    color = vec4(mix(currentColor, prevColor, 0.9), 0.0625);
    //color = vec4(mix(mix(currentColor, prevColor, 0.9), averageCurrentColor, velocityDisocclusion), 1.0);
}