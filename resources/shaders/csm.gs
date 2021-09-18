#version 410 core

// See https://community.khronos.org/t/cascaded-shadow-mapping-and-gl-texture-2d-array/71482
// See https://www.khronos.org/opengl/wiki/Geometry_Shader#Layered_rendering

layout (triangles) in;
// We are bringing in a single triangle (3 vertices) and outputting 4 triangles (12 vertices)
layout (triangle_strip, max_vertices = 12) out;

#define NUM_CASCADES 4

// Each cascaded shadow map has its own view-projection matrix
uniform mat4 shadowMatrices[NUM_CASCADES];

uniform int currentCascade = 0;

void main()
{
    int face = currentCascade;
    gl_Layer = face; // built-in variable that specifies to which face we render
    for(int i = 0; i < 3; ++i) { // for each triangle vertex
        //fsPosition = gl_in[i].gl_Position;
        vec4 fsPosition = gl_in[i].gl_Position;
        gl_Position = shadowMatrices[face] * fsPosition;
        EmitVertex(); // individual triangle vertex
    }
    EndPrimitive(); // triangle
}