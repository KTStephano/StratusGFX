STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

#include "common.glsl"
#include "alpha_test.glsl"

smooth in vec4 fsPosition;
smooth in vec2 fsTexCoords;
flat in int fsDrawID;

flat in int fsDiffuseMapped;
flat in int fsNormalMapped;
flat in int fsMetallicRoughnessMapped;
flat in int fsEmissiveMapped;

in vec3 fsNormal;
/**
 * Tangent space -> world space
 */
in mat3 fsTbnMatrix;

uniform vec3 lightPos;
uniform float farPlane;

layout (location = 0) out vec4 gColor;
layout (location = 1) out vec4 gNormal;
// layout (location = 2) out float gMetallic;

vec3 calculateNormal(in Material material, in vec2 texCoords) {
    vec3 normal = texture(material.normalMap, texCoords).rgb;
    // Normals generally have values from [-1, 1], but inside
    // an OpenGL texture they are transformed to [0, 1]. To convert
    // them back, we multiply by 2 and subtract 1.
    normal = normalize(normal * 2.0 - vec3(1.0)); // [0, 1] -> [-1, 1]
    // fsTbnMatrix goes from tangent space (defined by coordinate system of normal map)
    // to object space, and then model no translate moves to world space without translating
    normal = normalize(fsTbnMatrix * normal);
    normal = (normal + vec3(1.0)) * 0.5; // [-1, 1] -> [0, 1]

    return normal;
}

void main() {
    Material material = materials[materialIndices[fsDrawID]];
    vec4 baseColor = bitwiseAndBool(material.flags, GPU_DIFFUSE_MAPPED) ? texture(material.diffuseMap, fsTexCoords) : FLOAT4_TO_VEC4(material.diffuseColor);

    runAlphaTest(baseColor.a);

    vec3 normal = bool(fsNormalMapped) ? calculateNormal(material, fsTexCoords) : (fsNormal + 1.0) * 0.5; // [-1, 1] -> [0, 1]

    float roughness = material.metallicRoughness[1];
    float metallic = material.metallicRoughness[0];
    vec2 metallicRoughness = bool(fsMetallicRoughnessMapped) ? texture(material.metallicRoughnessMap, fsTexCoords).bg : vec2(metallic, roughness);

    metallic = metallicRoughness.x;
    roughness = metallicRoughness.y;

    vec3 emissive = bool(fsEmissiveMapped) ? texture(material.emissiveMap, fsTexCoords).rgb : FLOAT3_TO_VEC3(material.emissiveColor);
    if (length(emissive) > 0) {
        baseColor = vec4(emissive, 1.0);
    }

    // get distance between fragment and light source
    float lightDistance = length(fsPosition.xyz - lightPos);
    
    // map to [0;1] range by dividing by far_plane
    lightDistance = saturate(lightDistance / farPlane);
    
    // write this as modified depth
    gl_FragDepth = lightDistance;

    gColor = vec4(baseColor.rgb, length(emissive));
    gNormal = vec4(normal, roughness);
    // gMetallic = metallic;
}