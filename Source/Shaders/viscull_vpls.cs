STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

// This defines a 1D local work group of 1 (y and z default to 1)
// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
// Also see https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction
layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

#include "pbr.glsl"
#include "vpl_common.glsl"

uniform vec3 infiniteLightColor;
uniform int totalNumLights;
uniform vec3 viewPosition;

// for vec2 with std140 it always begins on a 2*4 = 8 byte boundary
// for vec3, vec4 with std140 it always begins on a 4*4=16 byte boundary

// See https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Memory_layout
// for information on layout std140 and std430. In std140 floats aren't guaranteed
// to be packed and so float arrays in OpenGL are not the same as float arrays in C/C++.
//
// This changes with std430 where it enforces equivalency between OpenGL and C/C++ float arrays
// by tightly packing them.
layout (std430, binding = 0) readonly buffer inoutBlock1 {
    VplData lightData[];
};

layout (std430, binding = 4) buffer inoutBlock2 {
    VplData updatedLightData[];
};

// layout (std430, binding = 1) buffer outputBlock1 {
//     int numVisible;
// };

layout (std430, binding = 3) buffer outputBlock2 {
    int vplVisibleIndex[];
};

shared bool lightVisible[MAX_TOTAL_VPLS_BEFORE_CULLING];
shared int lightVisibleIndex;
shared ivec3 probeLookupTableDimensions;
// shared int localNumVisible;

void main() {
    int stepSize = int(gl_NumWorkGroups.x * gl_WorkGroupSize.x * gl_NumWorkGroups.y * gl_WorkGroupSize.y);

    if (gl_LocalInvocationIndex == 0) {
        probeLookupTableDimensions = imageSize(probeRayLookupTable);
    }

    barrier();

    // Set all visible flags to false
    for (int index = int(gl_LocalInvocationIndex); index < totalNumLights; index += stepSize) {
        lightVisible[index] = false;
    }

    barrier();

    for (int index = int(gl_LocalInvocationIndex); index < totalNumLights; index += stepSize) {
        lightVisible[index] = true;
        vec3 lightPos = lightData[index].position.xyz;
        vec3 cascadeBlends = vec3(dot(cascadePlanes[0], vec4(lightPos, 1.0)),
                                dot(cascadePlanes[1], vec4(lightPos, 1.0)),
                                dot(cascadePlanes[2], vec4(lightPos, 1.0)));
        float shadowFactor = 1.0 - calculateInfiniteShadowValue(vec4(lightPos, 1.0), cascadeBlends, infiniteLightDirection, false);
        if (shadowFactor < 1.0) {
            lightVisible[index] = true;
            // int next = atomicAdd(localNumVisible, 1);
            // vplVisibleIndex[next] = index;
            //lightData[index].color = vec4(vec3(1.0) * 500, 1.0);
        }
    }

    barrier();

    // if (gl_LocalInvocationIndex == 0) {
    //     lightVisibleIndex = 0;
    // }

    // barrier();

    // const int maxHelperThreads = 8;
    // if (gl_LocalInvocationIndex < maxHelperThreads) {
    //     for (int index = int(gl_LocalInvocationIndex); index < totalNumLights; index += maxHelperThreads) {
    //         int localIndex = atomicAdd(lightVisibleIndex, 1);
    //         if (localIndex > MAX_TOTAL_VPLS_PER_FRAME) {
    //             break;
    //         }
    //         updatedLightData[localIndex] = lightData[index];
    //         updatedLightData[localIndex].visible = lightVisible[index] ? 1 : 0;
    //         // + 1 since we store the count in the first slot
    //         vplVisibleIndex[localIndex + 1] = index;
    //     }
    // }

    // barrier();

    if (gl_LocalInvocationIndex == 0) {
        lightVisibleIndex = totalNumLights > MAX_TOTAL_VPLS_PER_FRAME ? MAX_TOTAL_VPLS_PER_FRAME : totalNumLights;
        vplVisibleIndex[0] = lightVisibleIndex;
    }

    barrier();

    for (int index = int(gl_LocalInvocationIndex); index < lightVisibleIndex; index += stepSize) {
        updatedLightData[index] = lightData[index];
        updatedLightData[index].visible = lightVisible[index] ? 1 : 0;
        // + 1 since we store the count in the first slot
        vplVisibleIndex[index + 1] = index;

        vec3 lightPos = lightData[index].position.xyz;
        writeProbeIndexToLookupTable(probeLookupTableDimensions, viewPosition, lightPos, index);
    }
}