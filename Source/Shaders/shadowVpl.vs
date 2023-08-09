STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

// Enables gl_Layer and gl_ViewportIndex in the vertex shader (no geometry shader required)
#extension GL_ARB_shader_viewport_layer_array : require

#include "mesh_data.glsl"
#include "common.glsl"

uniform mat4 shadowMatrix;

uniform int layer;

layout (std430, binding = 13) readonly buffer SSBO3 {
    mat4 modelMatrices[];
};

smooth out vec4 fsPosition;
smooth out vec2 fsTexCoords;
flat out int fsDrawID;

flat out int fsDiffuseMapped;
flat out int fsNormalMapped;
flat out int fsMetallicRoughnessMapped;
flat out int fsEmissiveMapped;

// Made using the tangent, bitangent and normal
out mat3 fsTbnMatrix;
out vec3 fsNormal;

void main() {
    Material material = materials[materialIndices[gl_DrawID]];
    uint flags = material.flags;

    fsDiffuseMapped = int(bitwiseAndBool(flags, GPU_DIFFUSE_MAPPED));
    fsNormalMapped = int(bitwiseAndBool(flags, GPU_NORMAL_MAPPED));
    fsMetallicRoughnessMapped = int(bitwiseAndBool(flags, GPU_METALLIC_ROUGHNESS_MAPPED));
    fsEmissiveMapped = int(bitwiseAndBool(flags, GPU_EMISSIVE_MAPPED));

    // Select which layer of the depth texture we will write to
	// (DEPTH_LAYER is defined in C++ code)
	//gl_Layer = DEPTH_LAYER;
    gl_Layer = layer;

    fsDrawID = gl_DrawID;
    fsTexCoords = getTexCoord(gl_VertexID);
    fsPosition = modelMatrices[gl_DrawID] * vec4(getPosition(gl_VertexID), 1.0);

    mat3 modelNoTranslate = mat3(modelMatrices[gl_DrawID]);
    fsNormal = normalize(modelNoTranslate * getNormal(gl_VertexID));

    vec3 n = getNormal(gl_VertexID);
    vec3 t = getTangent(gl_VertexID);

    // re-orthogonalize T with respect to N - see end of https://learnopengl.com/Advanced-Lighting/Normal-Mapping
    // this is also called Graham-Schmidt
    t = normalize(t - dot(t, n) * n);

    // then retrieve perpendicular vector B and do the same
    //vec3 b = normalize(cross(n, t));
    vec3 b = getBitangent(gl_VertexID);
    b = normalize(b - dot(b, n) * n - dot(b, t) * t);
    fsTbnMatrix = modelNoTranslate * mat3(t, b, n);

    gl_Position = shadowMatrix * fsPosition;
}