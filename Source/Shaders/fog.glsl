STRATUS_GLSL_VERSION

uniform vec3 fogColor = vec3(0.5);
uniform float fogDensity = 0.0;

// For information about both general fog and half-space fog, see Chapter 8.5
// of "Foundations of Game Engine Development Volume 2 - Rendering"

// With fog the way it is implemented here is to approximate the light as it moves
// along a light path from a surface to the camera where the space in between is expected
// to contain participating media such as particles.
//
// Some portion of the light leaving the surface will make it to the camera. Some will be outscattered,
// and some will be absorbed.
//
// Some other portion of indirect light traveling along a different path will then be inscattered and subjected
// to the same possibilities as the direct light (could be absorbed, could be later outscattered, or it could make
// it to the camera).
//
// This is achieved with the following equation:
//      L0       = initial illuminance
//      Lambient = indirect contribution, here approximated as a fog color
//      distance = distance from surface to camera in world space
//      F        = fraction of incoming illuminance which is lost
//      Aab      = absorption coefficient = -ln(F)
//               --> -ln(F) since F is <= 1 so -ln(F) results in a positive value
//      Asc      = outscattering coefficient
//      Aex      = Aab + Asc
//      k        = fraction of scattered light directed towards the camera
//      Lin      = L0 * exp(-Aex * distance) + Lambient * ((k * Asc) / Aex) * (1.0 - exp(-Aex * distance))
//               --> Lambient * ((k*Asc) / Aex) are approximated by a final fog color which is input into the shader
//      f        = exp(-Aex * distance)
//      Lfinal   = L0 * f + fogColor * (1.0 - f)
//
// Since we combine multiple terms into single ones (ex: fogColor), it would be possible to require F, k, Asc and Lambient to
// be input and then we would compute Aab, Aex, and ((k * Asc) / Aex) automatically.
vec3 applyFog(vec3 color, float dist, float intensity) {
    float f = exp(-fogDensity * dist);
    vec3 fc = fogColor * intensity;
    return color * f + fc * (1.0 - f);
}