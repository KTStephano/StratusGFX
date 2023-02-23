STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

// This defines a 1D local work group of 1 (y and z default to 1)
// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
// Also see https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#include "pbr.glsl"
#include "vpl_tiled_deferred_culling.glsl"

uniform vec3 infiniteLightColor;

// for vec2 with std140 it always begins on a 2*4 = 8 byte boundary
// for vec3, vec4 with std140 it always begins on a 4*4=16 byte boundary

// See https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Memory_layout
// for information on layout std140 and std430. In std140 floats aren't guaranteed
// to be packed and so float arrays in OpenGL are not the same as float arrays in C/C++.
//
// This changes with std430 where it enforces equivalency between OpenGL and C/C++ float arrays
// by tightly packing them.
layout (std430, binding = 0) buffer vplLightData {
    float shadowFactors[];
};

layout (std430, binding = 4) buffer vplPositions {
    vec4 lightPositions[];
};

layout (std430, binding = 1) buffer numVisibleVPLs {
    int numVisible;
};

layout (std430, binding = 3) buffer outputBlock2 {
    int vplVisibleIndex[];
};

layout (std430, binding = 5) readonly buffer vplDiffuse {
    samplerCube diffuseCubeMaps[];
};

layout (std430, binding = 6) writeonly buffer vplColors {
    vec4 lightColors[];
};

layout (std430, binding = 7) readonly buffer vplIntensity {
    float lightIntensities[];
};

void main() {
    int index = int(gl_GlobalInvocationID.x);

    vec3 lightPos = lightPositions[index].xyz;
    vec3 cascadeBlends = vec3(dot(cascadePlanes[0], vec4(lightPos, 1.0)),
                              dot(cascadePlanes[1], vec4(lightPos, 1.0)),
                              dot(cascadePlanes[2], vec4(lightPos, 1.0)));
    float shadowFactor = 1.0 - calculateInfiniteShadowValue(vec4(lightPos, 1.0), cascadeBlends, infiniteLightDirection);
    if (shadowFactor < 0.99) {
        vec3 color = vec3(0.0);
        float offset = 0.3;
        float offsets[2] = float[](-offset, offset);
        // This should result in 2*2*2 = 8 samples
        for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 2; ++y) {
                for (int z = 0; z < 2; ++z) {
                    vec3 dirOffset = vec3(offsets[x], offsets[y], offsets[z]);
                    color += texture(diffuseCubeMaps[index], -infiniteLightDirection + dirOffset).rgb * infiniteLightColor;
                }
            }
        }

        int next = atomicAdd(numVisible, 1);
        shadowFactors[index] = shadowFactor;
        vplVisibleIndex[next] = index;
        lightColors[index] = vec4(color * lightIntensities[index], 1.0);
    }
}