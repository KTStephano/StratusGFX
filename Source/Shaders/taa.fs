// See this page: https://sugulee.wordpress.com/2021/06/21/temporal-anti-aliasingtaa-tutorial/

STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

in vec2 fsTexCoords;

uniform sampler2D screen;
uniform sampler2D prevScreen;
uniform sampler2D velocity;

out vec4 color;

void main() {
    vec2 velocityVal = texture(velocity, fsTexCoords).xy;
    vec2 prevTexCoords = fsTexCoords - velocityVal;

    vec3 currentColor = texture(screen, fsTexCoords).rgb;
    vec3 prevColor = texture(prevScreen, prevTexCoords).rgb;

    // Collect information around the texture coordinate and use it
    // to apply clamping (otherwise we get extreme ghosting)
    vec3 prevColor0 = textureOffset(screen, fsTexCoords, ivec2( 0,  1)).rgb;
    vec3 prevColor1 = textureOffset(screen, fsTexCoords, ivec2( 0, -1)).rgb;
    vec3 prevColor2 = textureOffset(screen, fsTexCoords, ivec2( 1,  0)).rgb;
    vec3 prevColor3 = textureOffset(screen, fsTexCoords, ivec2(-1,  0)).rgb;

    vec3 boxMin = min(currentColor, min(prevColor0, min(prevColor1, min(prevColor2, prevColor3))));
    vec3 boxMax = max(currentColor, max(prevColor0, max(prevColor1, max(prevColor2, prevColor3))));

    prevColor = clamp(prevColor, boxMin, boxMax);

    color = vec4(mix(currentColor, prevColor, 0.9), 1.0);
}