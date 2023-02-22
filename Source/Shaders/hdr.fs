STRATUS_GLSL_VERSION

in vec2 fsTexCoords;

uniform sampler2D screen;
uniform float gamma = 2.2;

out vec4 color;

vec3 saturate(vec3 x) {
    return clamp(x, 0.0, 1.0);
}

// See https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 applyACESFilm(vec3 color) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    // See https://community.khronos.org/t/saturate/53155 for saturate impl
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

vec3 applyReinhard(vec3 color) {
    return color / (color + vec3(1.0));
}

// The following were taken from https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/main/source/Renderer/shaders/tonemapping.glsl
// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
const mat3 ACESInputMat = mat3
(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777
);


// ODT_SAT => XYZ => D60_2_D65 => sRGB
const mat3 ACESOutputMat = mat3
(
    1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602
);

// ACES filmic tone map approximation
// see https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
// see https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/main/source/Renderer/shaders/tonemapping.glsl
vec3 RRTAndODTFit(vec3 color)
{
    vec3 a = color * (color + 0.0245786) - 0.000090537;
    vec3 b = color * (0.983729 * color + 0.4329510) + 0.238081;
    return a / b;
}


vec3 applyToneMapACES_Hill(vec3 color)
{
    color = ACESInputMat * color;

    // Apply RRT and ODT
    color = RRTAndODTFit(color);

    color = ACESOutputMat * color;

    // Clamp to [0, 1]
    color = clamp(color, 0.0, 1.0);

    return color;
}

// See http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 applyToneMap_Uncharted2(vec3 color)
{
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;
    return ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-E/F;
}

vec3 gammaCorrect(vec3 color) {
    return pow(color, vec3(1.0 / gamma));
}

// This uses either Reinhard or ACES tone mapping without exposure. More
// advanced techniques can be used to achieve a very different
// look and feel to the final output.
void main() {
    vec3 screenColor = texture(screen, fsTexCoords).rgb;

    // vec3 corrected = applyReinhard(screenColor);
    // corrected = gammaCorrect(corrected);

    vec3 corrected = applyACESFilm(screenColor);
    //vec3 corrected = applyToneMapACES_Hill(2.0 * screenColor);
    // vec3 corrected = applyToneMap_Uncharted2(2.0 * screenColor);
    // const float W = 11.2;
    // vec3 whiteScale = vec3(1.0) / applyToneMap_Uncharted2(vec3(W));
    // corrected = corrected * whiteScale;
    // Gamma correction
    corrected = gammaCorrect(corrected);

    color = vec4(corrected, 1.0);
}