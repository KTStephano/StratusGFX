STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

in vec3 fsTexCoords;

layout (location = 0) out vec4 fsColor;
#ifdef VPL_PIPELINE
layout (location = 1) out vec4 fsPosition;
uniform vec3 vplLocation;
#else
layout (location = 1) out vec2 fsVelocity;
#endif

uniform samplerCube skybox;
uniform vec3 colorMask = vec3(1.0);
uniform float intensity = 0.0;

void main() {
    fsColor = vec4(5.0 * intensity * vec3(colorMask) * texture(skybox, fsTexCoords).rgb, 1.0);

    #ifdef VPL_PIPELINE
    // TODO: replace addition with world up vector * scale
    fsPosition = vec4(vplLocation + vec3(0.0, 300.0, 0.0), 1.0);
    #else
    fsVelocity = vec2(0.0);
    #endif
}