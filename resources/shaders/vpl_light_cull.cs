STRATUS_GLSL_VERSION

// This defines a 1D local work group of 1 (y and z default to 1)
// See the Compute section of the OpenGL Superbible for more information
//
// Also see https://medium.com/@daniel.coady/compute-shaders-in-opengl-4-3-d1c741998c03
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// for vec2 with std140 it always begins on a 2*4 = 8 byte boundary
// for vec3, vec4 with std140 it always begins on a 4*4=16 byte boundary

layout (std430, binding = 1) buffer block2 {
    float outputArray[];
};

void main() {
    int index = int(gl_GlobalInvocationID.x);
    outputArray[index] = (index + 1) * 4.0;
}