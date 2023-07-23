// Anti-Aliasing common

STRATUS_GLSL_VERSION

// Notice how the green component is favored most heavily since perceptually
// it tends to contribute most to the luminance of a color
float linearColorToLuminance(in vec3 linearColor) {
    return 0.2126 * linearColor.r + 0.7152 * linearColor.g + 0.0722 * linearColor.b;
}
	
vec3 luminanceColorAdjust(in vec3 color) {
    float luminance = linearColorToLuminance(color);
    float luminanceWeight = 1.0 / (1.0 + luminance);
    return color * luminanceWeight;
}

vec3 tonemap(in vec3 color) {
    // Reinhard
    return color / (color + vec3(1.0));
}

vec3 inverseTonemap(in vec3 color) {
    return color / (vec3(1.0) - color);
}