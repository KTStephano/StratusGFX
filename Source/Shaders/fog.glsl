STRATUS_GLSL_VERSION

uniform vec3 fogColor = vec3(0.5);
uniform float fogDensity = 0.0;

// For information about both general fog and half-space fog, see Chapter 8.5
// of "Foundations of Game Engine Development Volume 2 - Rendering"

vec3 applyFog(vec3 color, float dist, float intensity) {
    float f = exp(-fogDensity * dist);
    //vec3 fc = vec3(79 / 255.0, 105 / 255.0, 136 / 255.0) * intensity;
    vec3 fc = fogColor * intensity;
    return color * f + fc * (1.0 - f);
}