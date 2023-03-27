STRATUS_GLSL_VERSION

// Matches the definition in StratusGpuCommon.h
// We use float arrays to get around padding requirements
// which pad vec3 to vec4 (see Graphics Rendering Cookbook, programmable vertex pulling)
struct MeshData {
    float position[3];
    float texCoord[2];
    float normal[3];
    float tangent[3];
    float bitangent[3];
};

layout (std430, binding = 32) readonly buffer MeshDataSSBO {
    MeshData meshData[];
};

vec3 getPosition(uint i) {
    return vec3(meshData[i].position[0], meshData[i].position[1], meshData[i].position[2]);
}

vec2 getTexCoord(uint i) {
    return vec2(meshData[i].texCoord[0], meshData[i].texCoord[1]);
}

vec3 getNormal(uint i) {
    return vec3(meshData[i].normal[0], meshData[i].normal[1], meshData[i].normal[2]);
}

vec3 getTangent(uint i) {
    return vec3(meshData[i].tangent[0], meshData[i].tangent[1], meshData[i].tangent[2]);
}

vec3 getBitangent(uint i) {
    return vec3(meshData[i].bitangent[0], meshData[i].bitangent[1], meshData[i].bitangent[2]);
}