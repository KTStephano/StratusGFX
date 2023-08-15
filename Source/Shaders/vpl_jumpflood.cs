STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

// This defines a 1D local work group of 1 (y and z default to 1)
// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
// Also see https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction
layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

#include "pbr.glsl"
#include "common.glsl"
#include "vpl_common.glsl"

uniform vec3 viewPosition;

uniform int probeGridStepSize;

layout (r32i) readonly uniform iimage3D probeRayLookupTableReadonly;

// for vec2 with std140 it always begins on a 2*4 = 8 byte boundary
// for vec3, vec4 with std140 it always begins on a 4*4=16 byte boundary

// See https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Memory_layout
// for information on layout std140 and std430. In std140 floats aren't guaranteed
// to be packed and so float arrays in OpenGL are not the same as float arrays in C/C++.
//
// This changes with std430 where it enforces equivalency between OpenGL and C/C++ float arrays
// by tightly packing them.
layout (std430, binding = 0) readonly buffer inputBlock1 {
    VplData lightData[];
};

// layout (std430, binding = 1) buffer outputBlock1 {
//     int numVisible;
// };

layout (std430, binding = 1) readonly buffer inputBlock2 {
    int vplVisibleIndex[];
};

shared ivec3 probeLookupTableDimensions;
// shared int localNumVisible;

const int offsets[] = int[](
    -1, 0, 1
);

void main() {
    if (gl_LocalInvocationIndex == 0) {
        probeLookupTableDimensions = imageSize(probeRayLookupTable);
    }

    barrier();

    // Jump Flood Algorithm (JFA) to fill the grid entirely

    if (probeLookupTableDimensions.x != probeLookupTableDimensions.y || probeLookupTableDimensions.x != probeLookupTableDimensions.z) {
        return;
    }

    int stepSizeX = int(gl_NumWorkGroups.x * gl_WorkGroupSize.x);
    int stepSizeY = int(gl_NumWorkGroups.y * gl_WorkGroupSize.y);
    int stepSizeZ = int(gl_NumWorkGroups.z * gl_WorkGroupSize.z);

    int xindex = int(gl_GlobalInvocationID.x);
    int yindex = int(gl_GlobalInvocationID.y);
    int zindex = int(gl_GlobalInvocationID.z);

    // bool success;
    // ivec3 integerTableIndex;
    // computeProbeIndexFromPosition(probeRayLookupDimensions, viewPosition, worldPos, success, integerTableIndex);

    ivec3 baseProbeIndex = ivec3(xindex, yindex, zindex);
    vec3 baseWorldPos = probeIndexToWorldPos(probeLookupTableDimensions, viewPosition, baseProbeIndex);

    int readLightIndex = int(imageLoad(probeRayLookupTableReadonly, baseProbeIndex).r);

    float bestDistance = FLOAT_MAX;
    if (readLightIndex >= 0) {
        vec3 lightPos = lightData[readLightIndex].position.xyz;
        bestDistance = length(baseWorldPos - lightPos);
    }

    // vec3 lightPos = lightData[readLightIndex].position.xyz;

    for (int i = 0; i < 3; ++i) {
        int oi = offsets[i] * probeGridStepSize;

        for (int j = 0; j < 3; ++j) {
            int oj = offsets[j] * probeGridStepSize;

            for (int k = 0; k < 3; ++k) {
                int ok = offsets[k] * probeGridStepSize;

                if (oi == 0 && oj == 0 && ok == 0) {
                    continue;
                }

                ivec3 probeIndex = baseProbeIndex + ivec3(oi, oj, ok);
                int newReadLightIndex = int(imageLoad(probeRayLookupTableReadonly, probeIndex));

                if (newReadLightIndex < 0) {
                    continue;
                }

                //vec3 pos = probeIndexToWorldPos(probeLookupTableDimensions, viewPosition, probeIndex);

                vec3 lightPos = lightData[newReadLightIndex].position.xyz;
                float newDistance = length(baseWorldPos - lightPos);

                if (newDistance < bestDistance) {
                    readLightIndex = newReadLightIndex;
                    bestDistance = newDistance;
                }
            }
        }
    }

    if (readLightIndex >= 0) {
        writeProbeIndexToLookupTableWithBoundsCheck(probeLookupTableDimensions, baseProbeIndex, readLightIndex);
    }
}