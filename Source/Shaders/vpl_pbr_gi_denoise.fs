STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "pbr.glsl"
#include "pbr2.glsl"
#include "vpl_common.glsl"

// Input from vertex shader
in vec2 fsTexCoords;
out vec3 color;

// in/out frame texture
uniform sampler2D screen;
uniform sampler2D indirectIllumination;

void main() {
    vec3 screenColor = texture(screen, fsTexCoords).rgb;

    vec3 centerIllum = texture(indirectIllumination, fsTexCoords).rgb;
    //vec3 centerIllum = texture(indirectIllumination, fsTexCoords).rgb;
    vec3 topIllum    = textureOffset(indirectIllumination, fsTexCoords, ivec2( 0,  1)).rgb;
    vec3 botIllum    = textureOffset(indirectIllumination, fsTexCoords, ivec2( 0, -1)).rgb;
    vec3 rightIllum  = textureOffset(indirectIllumination, fsTexCoords, ivec2( 1,  0)).rgb;
    vec3 leftIllum   = textureOffset(indirectIllumination, fsTexCoords, ivec2(-1,  0)).rgb;
    
    //vec3 illumAvg = (centerIllum + topIllum + botIllum + rightIllum + leftIllum) / 6.0;
    vec3 illumAvg = (centerIllum + topIllum + botIllum + rightIllum + leftIllum) / 5.0;

    //vec3 illumAvg = texture(indirectIllumination, fsTexCoords).rgb;

    color = screenColor + illumAvg;
}