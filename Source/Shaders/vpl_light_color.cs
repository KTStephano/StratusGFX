STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

// This defines a 1D local work group of 1 (y and z default to 1)
// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
// Also see https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction
layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

#include "pbr.glsl"
#include "vpl_common.glsl"

uniform vec3 infiniteLightColor;

// for vec2 with std140 it always begins on a 2*4 = 8 byte boundary
// for vec3, vec4 with std140 it always begins on a 4*4=16 byte boundary

// See https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Memory_layout
// for information on layout std140 and std430. In std140 floats aren't guaranteed
// to be packed and so float arrays in OpenGL are not the same as float arrays in C/C++.
//
// This changes with std430 where it enforces equivalency between OpenGL and C/C++ float arrays
// by tightly packing them.
layout (std430, binding = 0) buffer inoutBlock1 {
    VplData lightData[];
};

layout (std430, binding = 1) readonly buffer inputBlock1 {
    int numVisible;
};

layout (std430, binding = 3) readonly buffer inputBlock2 {
    int vplVisibleIndex[];
};

uniform samplerCubeArray diffuseCubeMaps;

layout (std430, binding = 4) readonly buffer inputBlock3 {
    int diffuseIndices[];
};

void main() {
    int stepSize = int(gl_NumWorkGroups.x * gl_WorkGroupSize.x);

    for (int i = int(gl_GlobalInvocationID.x); i < numVisible; i += stepSize) {
        int index = vplVisibleIndex[i];
        VplData data = lightData[index];
        float diffuseIndex = float(diffuseIndices[index]);
        // First two samples from the exact direction vector for a total of 10 samples after loop
        vec3 color = 2.0 * textureLod(diffuseCubeMaps, vec4(-infiniteLightDirection, diffuseIndex), 0).rgb * infiniteLightColor;
        float offset = 0.5;
        float offsets[2] = float[](-offset, offset);
        // This should result in 2*2*2 = 8 samples, + 2 from above = 10
        for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 2; ++y) {
                for (int z = 0; z < 2; ++z) {
                    vec3 dirOffset = vec3(offsets[x], offsets[y], offsets[z]);
                    color += textureLod(diffuseCubeMaps, vec4(-infiniteLightDirection + dirOffset, diffuseIndex), 0).rgb * infiniteLightColor;
                }
            }
        }

        lightData[index].color = vec4(color * data.intensity, 1.0);
    }
}