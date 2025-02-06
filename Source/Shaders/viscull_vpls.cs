STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "vpl_common.glsl"
#include "bindings.glsl"

uniform int totalNumLights;

// This defines a 1D local work group of 1 (y and z default to 1)
// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
// Also see https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction
layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

// for vec2 with std140 it always begins on a 2*4 = 8 byte boundary
// for vec3, vec4 with std140 it always begins on a 4*4=16 byte boundary

// See https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Memory_layout
// for information on layout std140 and std430. In std140 floats aren't guaranteed
// to be packed and so float arrays in OpenGL are not the same as float arrays in C/C++.
//
// This changes with std430 where it enforces equivalency between OpenGL and C/C++ float arrays
// by tightly packing them.
layout (std430, binding = VPL_PROBE_DATA_BINDING) readonly buffer inputBlock1 {
    VplData probes[];
};

//layout (std430, binding = VPL_PROBE_CONTRIB_BINDING) readonly buffer inoutBlock2 {
//    int vplContributionFlags[];
//};

// layout (std430, binding = 1) buffer outputBlock1 {
//     int numVisible;
// };

layout (std430, binding = VPL_PROBE_INDICES_BINDING) coherent buffer outputBlock1 {
    int vplVisibleIndex[];
};

layout (std430, binding = VPL_PROBE_INDEX_COUNTERS_BINDING) coherent buffer outputBlock2 {
    int vplVisibleIndexCounters[];
};

shared int numPresent;

void main() {
    // One dispatch per bucket
    int baseBucketIndex = int(gl_WorkGroupID.x);
    int offsetBucketIndex = computeOffsetBucketIndex(baseBucketIndex);
    
    // Clear counter
    if (gl_LocalInvocationIndex == 0) {
        numPresent = 0;
    }

    barrier();

    int stepSize = int(gl_WorkGroupSize.x*gl_WorkGroupSize.y*gl_WorkGroupSize.z);

    for (int index = int(gl_LocalInvocationIndex); index < totalNumLights; index += stepSize) {
        VplData probe = probes[index];
        //if (probe.activeProbe > 0) {
            ivec3 bucketCoords = computeBaseBucketCoords(probe.position.xyz);
            if (!baseBucketCoordsWithinRange(bucketCoords)) {
                continue;
            }

            if (computeBaseBucketIndex(bucketCoords) != baseBucketIndex) {
                // Skip since it doesn't match our bucket
                continue;
            }

            int localIndex = atomicAdd(numPresent, 1);
            if (localIndex >= MAX_VPLS_PER_BUCKET) {
                // Ran out of slots for probes
                break;
            }

            vplVisibleIndex[offsetBucketIndex + localIndex] = index;
        //}
    }

    barrier();

    if (gl_LocalInvocationID == 0) {
        if (numPresent > MAX_VPLS_PER_BUCKET) {
            numPresent = MAX_VPLS_PER_BUCKET;
        }

        vplVisibleIndexCounters[baseBucketIndex] = numPresent;
    }

    memoryBarrierBuffer();
}