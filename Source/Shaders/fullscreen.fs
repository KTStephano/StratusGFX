STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

in vec2 fsTexCoords;

uniform sampler2D screen;

// Used to copy the final depth to its hiz buffer output
uniform sampler2D inputDepth;
layout (r32f) writeonly uniform image2D outputDepth;

out vec4 color;

void main() {
    // float depth = textureLod(inputDepth, fsTexCoords, 0).r;
    // ivec2 pixelCoords = ivec2(fsTexCoords * vec2(textureSize(inputDepth, 0).xy));

    // imageStore(outputDepth, pixelCoords, vec4(depth));

    color = vec4(texture(screen, fsTexCoords).rgb, 1.0);
}