STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "vpl_common.glsl"

uniform int totalNumLights;

// This defines a 1D local work group of 1 (y and z default to 1)
// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
// Also see https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction
layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// for vec2 with std140 it always begins on a 2*4 = 8 byte boundary
// for vec3, vec4 with std140 it always begins on a 4*4=16 byte boundary

// See https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Memory_layout
// for information on layout std140 and std430. In std140 floats aren't guaranteed
// to be packed and so float arrays in OpenGL are not the same as float arrays in C/C++.
//
// This changes with std430 where it enforces equivalency between OpenGL and C/C++ float arrays
// by tightly packing them.
layout (std430, binding = 0) readonly buffer inoutBlock1 {
    int vplContributionFlags[];
};

// layout (std430, binding = 1) buffer outputBlock1 {
//     int numVisible;
// };

layout (std430, binding = 1) writeonly buffer outputBlock1 {
    int vplVisibleIndex[];
};

shared int indexMarker;

void main() {
    int stepSize = int(gl_NumWorkGroups.x * gl_WorkGroupSize.x);

    if (gl_LocalInvocationID == 0) {
        indexMarker = 0;
    }

    barrier();

    for (int index = int(gl_LocalInvocationIndex); index < totalNumLights; index += stepSize) {
        if (vplContributionFlags[index] > 0) {
            int localIndex = atomicAdd(indexMarker, 1);
            if (localIndex > MAX_TOTAL_VPLS_PER_FRAME) {
                break;
            }

            // + 1 since we store the count in the first slot
            vplVisibleIndex[localIndex + 1] = index;
        }
    }

    barrier();

    if (gl_LocalInvocationIndex == 0) {
        indexMarker = indexMarker > MAX_TOTAL_VPLS_PER_FRAME ? MAX_TOTAL_VPLS_PER_FRAME : indexMarker;
        vplVisibleIndex[0] = indexMarker;
    }
}