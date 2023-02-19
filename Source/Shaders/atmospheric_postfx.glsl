STRATUS_GLSL_VERSION

// This is based on something called "directional median filter." This is the process of taking
// a few samples (3 here) along a certain direction (clip space light direction here) and computing
// the median value of the set of samples taken.
float getAtmosphericIntensity(sampler2DRect atmosphere, vec3 lightPosition, vec2 pixelCoords) {
    // Remember that lightPosition.z contains 2*Lz
    vec2 center = pixelCoords;
    vec2 direction = normalize(lightPosition.xy - center * lightPosition.z);
    float a = texture(atmosphere, center).x;
    float b = texture(atmosphere, center + direction).x;
    float c = texture(atmosphere, center - direction).x;
    //float d = texture(atmosphere, center + 2 * direction).x;
    //float e = texture(atmosphere, center - 2 * direction).x;
    // See page 354, eq. 10.83
    return min(max(min(a, b), c), max(a, b));
    //return max(a, max(b, max(c, max(d, e))));
    //return (a + b + c) / 3.0;
}