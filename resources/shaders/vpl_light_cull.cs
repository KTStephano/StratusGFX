STRATUS_GLSL_VERSION

// This defines a 1D local work group of 1 (y and z default to 1)
// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// for vec2 with std140 it always begins on a 2*4 = 8 byte boundary
// for vec3, vec4 with std140 it always begins on a 4*4=16 byte boundary

// See https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Memory_layout
// for information on layout std140 and std430. In std140 floats aren't guaranteed
// to be packed and so float arrays in OpenGL are not the same as float arrays in C/C++.
//
// This changes with std430 where it enforces equivalency between OpenGL and C/C++ float arrays
// by tightly packing them.
layout (std430, binding = 1) buffer block2 {
    float outputArray[];
};

uniform vec3 infiniteLightDirection;
uniform sampler2DArrayShadow infiniteLightShadowMap;
// Each vec4 offset has two pairs of two (x, y) texel offsets. For each cascade we sample
// a neighborhood of 4 texels and additive blend the results.
uniform vec4 shadowOffset[2];
// Represents a plane which transitions from 0 to 1 as soon as two cascades overlap
uniform vec4 cascadePlanes[3];
uniform mat4 cascadeProjViews[4];

uniform int numLights;

void main() {
    int index = int(gl_GlobalInvocationID.x);
    outputArray[index] = (index + 1) * 4.0;
}