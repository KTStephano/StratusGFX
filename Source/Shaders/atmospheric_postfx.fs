STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "common.glsl"
#include "atmospheric_postfx.glsl"

uniform sampler2DRect atmosphereBuffer;
// Holds color values after lighting has been applied
uniform sampler2D screenBuffer;
// Contains 2*Lz*(Xlight, Ylight, 1) -- see page 354, eq. 10.81 and page 355, listing 10.14
uniform vec3 lightPosition;
uniform vec3 lightColor;

smooth in vec2 fsTexCoords;

out vec4 color;

void main() {
    vec2 widthHeight = textureSize(atmosphereBuffer).xy;
    vec3 screenColor = texture(screenBuffer, fsTexCoords).rgb;
    vec3 atmosphereColor = normalize(lightColor); //10 * screenColor + lightColor / 2.0;
    float intensity = getAtmosphericIntensity(atmosphereBuffer, lightPosition, fsTexCoords * widthHeight);

    color = vec4(screenColor + intensity * atmosphereColor, 1.0);
}