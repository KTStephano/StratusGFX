// Copies top level depth into the 32-bit float texture to make it
// easier to do the rest in compute (can't write to depth textures
// from compute :()

STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require
#extension GL_ARB_sparse_texture2 : require

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

precision highp float;
precision highp int;
precision highp uimage2D;
precision highp sampler2D;
precision highp sampler2DArrayShadow;

uniform sampler2D depthInput;
layout (r32f) writeonly uniform image2D depthOutput;

void main() {
    ivec2 size = imageSize(depthOutput).xy;

    ivec2 pixelXY = ivec2(gl_GlobalInvocationID.xy);

    if (pixelXY.x < size.x && pixelXY.y < size.y) {
        vec2 uv = (vec2(pixelXY) + 0.5) / vec2(size);

        float depth = textureLod(depthInput, uv, 0).r;

        imageStore(depthOutput, pixelXY, vec4(depth));
    }
}