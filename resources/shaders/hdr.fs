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
    // Gamma correction
    corrected = gammaCorrect(corrected);

    color = vec4(corrected, 1.0);
}