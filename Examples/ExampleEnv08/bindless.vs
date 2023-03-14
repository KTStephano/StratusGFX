STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

// This matches the C++ definition
struct VertexData {
    float position[3];
    float uv[2];
    float normal[3];
};

// readonly SSBO containing the data
layout(binding = 0, std430) readonly buffer ssbo1 {
    VertexData data[];
};

uniform mat4 projection;
uniform mat4 view;

// Helper functions to manually unpack the data into vectors given an index
vec3 getPosition(int index) {
    return vec3(
        data[index].position[0], 
        data[index].position[1], 
        data[index].position[2]
    );
}

vec2 getUV(int index) {
    return vec2(
        data[index].uv[0], 
        data[index].uv[1]
    );
}

vec3 getNormal(int index) {
    return vec3(
        data[index].normal[0], 
        data[index].normal[1], 
        data[index].normal[2]
    );
}

out vec2 fsUv;
out vec3 fsNormal;

void main()
{
    gl_Position = projection * view * vec4(getPosition(gl_VertexID), 1.0);

    fsUv = getUV(gl_VertexID);
    fsNormal = getNormal(gl_VertexID);
}
