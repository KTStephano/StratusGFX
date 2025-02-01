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
uniform float intensity = 3.0;

void main() {
    fsColor = intensity * vec4(colorMask, 1.0) * texture(skybox, fsTexCoords);
    #ifdef VPL_PIPELINE
    // TODO: replace addition with world up vector * scale
    fsPosition = vec4(vplLocation + vec3(0.0, 250.0, 0.0), 1.0);
    #else
    fsVelocity = vec2(0.0);
    #endif
}