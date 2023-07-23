STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

// This defines a 1D local work group of 1 (y and z default to 1)
// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
// Also see https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction
layout (local_size_x = 96, local_size_y = 1, local_size_z = 1) in;

// Each one specifies a different culling mode which has its own commands + model matrices
layout (std430, binding = 0) buffer ssbo1 {
    mat4 cull0PrevFrameModelMatrices[];
};

layout (std430, binding = 1) readonly buffer ssbo2 {
    mat4 cull0ModelMatrices[];
};

layout (std430, binding = 2) buffer ssbo3 {
    mat4 cull1PrevFrameModelMatrices[];
};

layout (std430, binding = 3) readonly buffer ssbo4 {
    mat4 cull1ModelMatrices[];
};

layout (std430, binding = 4) buffer ssbo5 {
    mat4 cull2PrevFrameModelMatrices[];
};

layout (std430, binding = 5) readonly buffer ssbo6 {
    mat4 cull2ModelMatrices[];
};

uniform int cull0NumMatrices;
uniform int cull1NumMatrices;
uniform int cull2NumMatrices;

void main() {
    int stepSize = int(gl_NumWorkGroups.x * gl_WorkGroupSize.x);

    // cull0
    for (int i = int(gl_LocalInvocationIndex); i < cull0NumMatrices; i += stepSize) {
        cull0PrevFrameModelMatrices[i] = cull0ModelMatrices[i];
    }

    // cull1
    for (int i = int(gl_LocalInvocationIndex); i < cull1NumMatrices; i += stepSize) {
        cull1PrevFrameModelMatrices[i] = cull1ModelMatrices[i];
    }

    // cull2
    for (int i = int(gl_LocalInvocationIndex); i < cull2NumMatrices; i += stepSize) {
        cull2PrevFrameModelMatrices[i] = cull2ModelMatrices[i];
    }
}