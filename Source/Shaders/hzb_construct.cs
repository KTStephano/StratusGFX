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

layout (r32f) readonly uniform image2D depthInput;
layout (r32f) writeonly uniform image2D depthOutput;

void main() {
    ivec2 sizeInput  = imageSize(depthInput).xy;
    ivec2 sizeOutput = imageSize(depthOutput).xy;

    ivec2 pixelXY = ivec2(gl_GlobalInvocationID.xy);
    ivec2 pixelXYInput = 2 * pixelXY;

    if (pixelXY.x < sizeOutput.x && pixelXY.y < sizeOutput.y) {
        ivec2 coord1 = pixelXYInput;
        ivec2 coord2 = pixelXYInput + ivec2(0, 1);
        ivec2 coord3 = pixelXYInput + ivec2(1, 0);
        ivec2 coord4 = pixelXYInput + ivec2(1, 1);

        float depth1 = imageLoad(depthInput, coord1).r;
        float depth2 = imageLoad(depthInput, coord2).r;
        float depth3 = imageLoad(depthInput, coord3).r;
        float depth4 = imageLoad(depthInput, coord4).r;

        float depth = min(min(depth1, depth2), min(depth3, depth4));

        imageStore(depthOutput, pixelXY, vec4(depth));
    }
}