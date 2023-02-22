// No longer used! Kept for reference purposes. See http://www.joshbarczak.com/blog/?p=667

STRATUS_GLSL_VERSION

layout (triangles) in;
// We are bringing in a single triangle (3 vertices) and outputting 6 triangles (18 vertices)
layout (triangle_strip, max_vertices = 18) out;

// Each cube map face has its own transform matrix
uniform mat4 shadowMatrices[6];

out vec4 fsPosition;

void main()
{
    for(int face = 0; face < 6; ++face) {
        gl_Layer = face; // built-in variable that specifies to which face we render
        for(int i = 0; i < 3; ++i) { // for each triangle vertex
            fsPosition = gl_in[i].gl_Position;
            gl_Position = shadowMatrices[face] * fsPosition;
            EmitVertex(); // individual triangle vertex
        }
        EndPrimitive(); // triangle
    }
}