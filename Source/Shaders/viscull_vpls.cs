STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "vpl_common.glsl"
#include "bindings.glsl"
#include "aabb.glsl"

uniform int totalNumLights;
uniform vec3 viewPosition;
uniform vec4 frustumPlanes[6];

// This defines a 1D local work group of 1 (y and z default to 1)
// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
// Also see https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction
layout (local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

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
shared int numWithinShortRange;
shared vec3 worldSpaceBucketCenter;
shared int baseBucketIndex;
shared int offsetBucketIndex;
shared bool bucketIsVisible;

void main() {
    if (gl_LocalInvocationIndex == 0) {
        ivec3 baseBucketCoords = ivec3(
            int(gl_WorkGroupID.x),
            int(gl_WorkGroupID.y),
            int(gl_WorkGroupID.z)
        );

        // baseBucketCoords are always positive, but relative coords go from [-max, +max)
        ivec3 relativeBucketCoords = baseBucketCoords - ivec3(HALF_VPL_BUCKETS_PER_DIM);

        // One dispatch per bucket
        worldSpaceBucketCenter = computeWorldSpaceBucketCenter(relativeBucketCoords, viewPosition);
        baseBucketIndex = computeBaseBucketIndex(baseBucketCoords);
        offsetBucketIndex = computeOffsetBucketIndex(baseBucketIndex);
    
        // Clear counter(s)
        numPresent = 0;
        numWithinShortRange = 0;

        //vplVisibleIndexCounters[baseBucketIndex] = 0;

        vec3 worldSpaceMin = convertRelativeCoordsToWorldPosition(relativeBucketCoords, viewPosition);
        vec3 worldSpaceMax = worldSpaceMin + vec3(float(WORLD_SPACE_PER_VPL_BUCKET));

        AABB aabb;
        aabb.vmin = vec4(worldSpaceMin, 1.0);
        aabb.vmax = vec4(worldSpaceMax, 1.0);
        bucketIsVisible = isAabbVisible(frustumPlanes, aabb);

        if (!bucketIsVisible) {
            // Clear flag since we will return early
            vplVisibleIndexCounters[baseBucketIndex] = 0;
        }
    }

    barrier();

    if (!bucketIsVisible) {
        return;
    }

    int stepSize = int(gl_WorkGroupSize.x*gl_WorkGroupSize.y*gl_WorkGroupSize.z);

    const float expandedCutoffOffset = float(WORLD_SPACE_PER_VPL_BUCKET);

    float oldCutoffDistance = 0.0;
    float cutoffDistance = WORLD_SPACE_PER_VPL_BUCKET * 0.5 + expandedCutoffOffset;
    int maxVplsAllowed = MAX_VPLS_PER_BUCKET;

    #define PERFORM_PROBE_CULL                                                          \
        VplData probe = probes[index];                                                  \
        float dist = distance(FLOAT3_TO_VEC3(probe.position), worldSpaceBucketCenter);  \
        bool added = dist > oldCutoffDistance && dist <= cutoffDistance;                \
        if (added) {                                                                    \
            int localIndex = atomicAdd(numPresent, 1);                                  \
            if (localIndex >= maxVplsAllowed) {                                         \
                /* Ran out of slots for probes */                                       \
                break;                                                                  \
            }                                                                           \
            vplVisibleIndex[offsetBucketIndex + localIndex] = index;                    \
        }

    // Perform first pass that bins based on exact bucket
    for (int index = int(gl_LocalInvocationIndex); index < totalNumLights; index += stepSize) {
        PERFORM_PROBE_CULL

        // If we get here, it was out of range - check if it would be visible to an expanded search
        if (!added && dist < (cutoffDistance + expandedCutoffOffset)) {
            atomicAdd(numWithinShortRange, 1);
        }
    }

    barrier();

    // Expand search radius and try again if needed
    oldCutoffDistance = cutoffDistance;
    cutoffDistance += expandedCutoffOffset;
    if (numPresent < 128 && numWithinShortRange > 0) {
       maxVplsAllowed = min(numPresent + numWithinShortRange, 128);
       for (int index = int(gl_LocalInvocationIndex); index < totalNumLights; index += stepSize) {
           PERFORM_PROBE_CULL
       }
    }

    barrier();

    // // Final expand but with reduced max
    // oldCutoffDistance = cutoffDistance;
    // cutoffDistance += WORLD_SPACE_PER_VPL_BUCKET;
    // maxVplsAllowed = 128;
    // if (numPresent < 128) {
    //     for (int index = int(gl_LocalInvocationIndex); index < totalNumLights; index += stepSize) {
    //         PERFORM_PROBE_CULL
    //     }
    // }

    barrier();

    if (gl_LocalInvocationID == 0) {
        if (numPresent > MAX_VPLS_PER_BUCKET) {
            numPresent = MAX_VPLS_PER_BUCKET;
        }

        vplVisibleIndexCounters[baseBucketIndex] = numPresent;
    }

    memoryBarrierBuffer();
}