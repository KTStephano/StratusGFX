STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "pbr.glsl"
#include "pbr2.glsl"
#include "vpl_common.glsl"

// Input from vertex shader
in vec2 fsTexCoords;

out vec3 combinedColor;
out vec3 giColor;

// in/out frame texture
uniform sampler2D screen;
uniform sampler2D velocity;
uniform sampler2D normal;
uniform sampler2D depth;

uniform sampler2D prevNormal;
uniform sampler2D prevDepth;

uniform sampler2D indirectIllumination;
uniform sampler2D prevIndirectIllumination;


void main() {
    vec3 screenColor = texture(screen, fsTexCoords).rgb;

    vec3 centerIllum = texture(indirectIllumination, fsTexCoords).rgb;
    vec3 topIllum    = textureOffset(indirectIllumination, fsTexCoords, ivec2( 0,  1)).rgb;
    vec3 botIllum    = textureOffset(indirectIllumination, fsTexCoords, ivec2( 0, -1)).rgb;
    vec3 rightIllum  = textureOffset(indirectIllumination, fsTexCoords, ivec2( 1,  0)).rgb;
    vec3 leftIllum   = textureOffset(indirectIllumination, fsTexCoords, ivec2(-1,  0)).rgb;

    vec3 gi = (centerIllum + topIllum + botIllum + rightIllum + leftIllum) / 5.0;
    vec3 prevGi = texture(prevIndirectIllumination, fsTexCoords).rgb;

    vec3 illumAvg = gi;
    //vec3 illumAvg = mix(prevGi, gi, 0.1);

    combinedColor = screenColor + illumAvg;
    giColor = illumAvg;
}

// void main() {
//     vec3 screenColor = texture(screen, fsTexCoords).rgb;

//     vec3 centerIllum = texture(indirectIllumination, fsTexCoords).rgb;
//     //vec3 centerIllum = texture(indirectIllumination, fsTexCoords).rgb;
//     vec3 topIllum    = textureOffset(indirectIllumination, fsTexCoords, ivec2( 0,  1)).rgb;
//     vec3 botIllum    = textureOffset(indirectIllumination, fsTexCoords, ivec2( 0, -1)).rgb;
//     vec3 rightIllum  = textureOffset(indirectIllumination, fsTexCoords, ivec2( 1,  0)).rgb;
//     vec3 leftIllum   = textureOffset(indirectIllumination, fsTexCoords, ivec2(-1,  0)).rgb;
    
//     //vec3 illumAvg = (centerIllum + topIllum + botIllum + rightIllum + leftIllum) / 6.0;
//     vec3 illumAvg = (centerIllum + topIllum + botIllum + rightIllum + leftIllum) / 5.0;

//     //vec3 illumAvg = texture(indirectIllumination, fsTexCoords).rgb;

//     combinedColor = screenColor + illumAvg;
//     giColor = illumAvg;
// }