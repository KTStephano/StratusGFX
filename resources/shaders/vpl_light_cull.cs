STRATUS_GLSL_VERSION

// This defines a 1D local work group of 1 (y and z default to 1)
// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#include "pbr.glsl"
#include "vpl_tiled_deferred_culling.glsl"

uniform vec3 viewPosition;

// for vec2 with std140 it always begins on a 2*4 = 8 byte boundary
// for vec3, vec4 with std140 it always begins on a 4*4=16 byte boundary

// See https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Memory_layout
// for information on layout std140 and std430. In std140 floats aren't guaranteed
// to be packed and so float arrays in OpenGL are not the same as float arrays in C/C++.
//
// This changes with std430 where it enforces equivalency between OpenGL and C/C++ float arrays
// by tightly packing them.
layout (std430, binding = 0) buffer vplLightData {
    VirtualPointLight lightData[];
};

layout (std430, binding = 1) buffer numVisibleVPLs {
    int numVisible;
};

layout (std430, binding = 3) buffer outputBlock2 {
    int vplVisibleIndex[];
};

void main() {
    int index = int(gl_GlobalInvocationID.x);
    VirtualPointLight vpl = lightData[index];
    vec3 lightPos = vpl.lightPosition.xyz;
    vec3 cascadeBlends = vec3(dot(cascadePlanes[0], vec4(lightPos, 1.0)),
                              dot(cascadePlanes[1], vec4(lightPos, 1.0)),
                              dot(cascadePlanes[2], vec4(lightPos, 1.0)));
    float shadowFactor = 1.0 - calculateInfiniteShadowValue(vec4(lightPos, 1.0), cascadeBlends, infiniteLightDirection);
    if (shadowFactor < 0.95) {
        int next = atomicAdd(numVisible, 1);
        vpl.distToCamera = length(lightPos - viewPosition);
        vpl.shadowFactor = shadowFactor;
        vplVisibleIndex[next] = index;
    }
}