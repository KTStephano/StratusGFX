STRATUS_GLSL_VERSION

// Important Papers:
//      -> SVGF: https://research.nvidia.com/sites/default/files/pubs/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A//svgf_preprint.pdf
//      -> ASVGF: https://cg.ivd.kit.edu/publications/2018/adaptive_temporal_filtering/adaptive_temporal_filtering.pdf
//      -> Q2RTX: https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s91046-real-time-path-tracing-and-denoising-in-quake-2.pdf
//      -> Q2RTX + Albedo Demodulation: https://cg.informatik.uni-freiburg.de/intern/seminar/raytracing%20-%20Keller%20-%20SIGGRAPH%202019%204%20Path%20Tracing.pdf
//      -> Reconstruction Filters: https://cg.informatik.uni-freiburg.de/intern/seminar/raytracing%20-%20Keller%20-%20SIGGRAPH%202019%204%20Path%20Tracing.pdf

#extension GL_ARB_bindless_texture : require

#include "pbr.glsl"
#include "pbr2.glsl"
#include "vpl_common.glsl"
#include "aa_common.glsl"

// Input from vertex shader
in vec2 fsTexCoords;

out vec3 combinedColor;
out vec3 giColor;

// in/out frame texture
uniform sampler2D screen;
uniform sampler2D albedo;
uniform sampler2D velocity;
uniform sampler2D normal;
uniform sampler2D depth;

uniform sampler2D prevNormal;
uniform sampler2D prevDepth;

uniform sampler2D indirectIllumination;
uniform sampler2D indirectShadows;
uniform sampler2D prevIndirectIllumination;

#define COMPONENT_WISE_MIN_VALUE 0.001

void main() {
    vec3 screenColor = texture(screen, fsTexCoords).rgb;
    vec2 velocityVal = texture(velocity, fsTexCoords).xy;
    vec2 prevTexCoords = fsTexCoords - velocityVal;
    //vec3 baseColor = texture(albedo, fsTexCoords).rgb;

    vec3 centerIllum = texture(indirectIllumination, fsTexCoords).rgb;
    // vec3 topIllum    = textureOffset(indirectIllumination, fsTexCoords, ivec2( 0,  1)).rgb;
    // vec3 botIllum    = textureOffset(indirectIllumination, fsTexCoords, ivec2( 0, -1)).rgb;
    // vec3 rightIllum  = textureOffset(indirectIllumination, fsTexCoords, ivec2( 1,  0)).rgb;
    // vec3 leftIllum   = textureOffset(indirectIllumination, fsTexCoords, ivec2(-1,  0)).rgb;

    // vec3 centerShadow = texture(indirectShadows, fsTexCoords).rgb;
    // vec3 topShadow    = textureOffset(indirectShadows, fsTexCoords, ivec2( 0,  1)).rgb;
    // vec3 botShadow    = textureOffset(indirectShadows, fsTexCoords, ivec2( 0, -1)).rgb;
    // vec3 rightShadow  = textureOffset(indirectShadows, fsTexCoords, ivec2( 1,  0)).rgb;
    // vec3 leftShadow   = textureOffset(indirectShadows, fsTexCoords, ivec2(-1,  0)).rgb;

    vec3 shadowFactor = vec3(0.0);
    float numShadowSamples = 0.0;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            ++numShadowSamples;
            shadowFactor += textureOffset(indirectShadows, fsTexCoords, ivec2(dx, dy)).rgb;
        }
    }
    shadowFactor /= numShadowSamples;

    //vec3 minColor = tonemap(min(centerIllum, min(topIllum, min(botIllum, min(rightIllum, leftIllum)))));
    //vec3 maxColor = tonemap(max(centerIllum, max(topIllum, max(botIllum, max(rightIllum, leftIllum)))));

    //vec3 gi = (centerIllum + topIllum + botIllum + rightIllum + leftIllum) / 5.0;
    vec3 gi = centerIllum * shadowFactor;
    //vec3 shadow = shadowFactor;
    //vec3 shadow = centerShadow;
    //gi = gi;
    // vec3 gi = centerIllum / (baseColor + PREVENT_DIV_BY_ZERO);
    //vec3 gi = centerIllum;

    // vec3 gi = vec3(0.0);
    // float numGiSamples = 0.0;
    // for (int dx = -1; dx <= 1; ++dx) {
    //     for (int dy = -1; dy <= 1; ++dy) {
    //         ++numGiSamples;
    //         //vec3 currAlbedo = textureOffset(albedo, fsTexCoords, ivec2(dx, dy)).rgb;
    //         vec3 illum = textureOffset(indirectIllumination, fsTexCoords, ivec2(dx, dy)).rgb;
    //         // currAlbedo = vec3(
    //         //     max(currAlbedo.r, COMPONENT_WISE_MIN_VALUE), 
    //         //     max(currAlbedo.g, COMPONENT_WISE_MIN_VALUE), 
    //         //     max(currAlbedo.b, COMPONENT_WISE_MIN_VALUE)
    //         // );
    //         gi += illum;
    //         //gi += vec3(luminance);
    //     }
    // }
    // gi /= numGiSamples;

    vec3 prevGi = texture(prevIndirectIllumination, prevTexCoords).rgb;
    // vec3 tmPrevGi = tonemap(prevGi);
    // tmPrevGi = vec3(
    //     clamp(tmPrevGi.r, minColor.r, maxColor.r),
    //     clamp(tmPrevGi.g, minColor.g, maxColor.g),
    //     clamp(tmPrevGi.b, minColor.b, maxColor.b)
    // );
    // prevGi = inverseTonemap(tmPrevGi);

    //vec3 illumAvg = centerIllum;
    vec3 illumAvg = gi;
    //vec3 illumAvg = shadowFactor;
    // a is effectively how many frames we can accumulate. So for example 1 / 0.1 = 10,
    // 1 / 0.05 = 20, etc.
    //vec3 illumAvg = mix(prevGi, gi, a);
    //vec3 illumAvg = mix(prevGi, gi, 0.1);
    //vec3 illumAvg = vec3(difference);

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