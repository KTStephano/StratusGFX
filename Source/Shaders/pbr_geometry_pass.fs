STRATUS_GLSL_VERSION

#extension GL_ARB_bindless_texture : require

//layout (early_fragment_tests) in;

#include "common.glsl"
#include "alpha_test.glsl"

//uniform float fsShininessVals[MAX_INSTANCES];
//uniform float fsShininess = 0.0;
uniform float heightScale = 0.1;

/**
 * Information about the camera
 */
uniform vec3 viewPosition;

uniform float emissiveTextureMultiplier = 1.0;

/**
 * Fragment information. All values should be
 * in world space.
 */
smooth in vec3 fsPosition;
//in vec3 fsViewSpacePos;
in vec3 fsNormal;
smooth in vec2 fsTexCoords;
in mat4 fsModel;
in mat3 fsModelNoTranslate;
flat in int fsDrawID;

/**
 * Tangent space -> world space
 */
in mat3 fsTbnMatrix;

in vec4 fsCurrentClipPos;
in vec4 fsPrevClipPos;

flat in int fsDiffuseMapped;
flat in int fsNormalMapped;
flat in int fsMetallicMapped;
flat in int fsRoughnessMapped;
flat in int fsMetallicRoughnessMapped;
flat in int fsEmissiveMapped;

// GBuffer outputs
//layout (location = 0) out vec3 gPosition;
layout (location = 0) out vec3 gNormal;
layout (location = 1) out vec4 gAlbedo;
layout (location = 2) out vec2 gReflectivityEmissive;
layout (location = 3) out vec3 gRoughnessMetallicAmbient;
// The structure buffer contains information related to depth in camera space. Useful for things such as ambient occlusion
// and atmospheric shadowing.
layout (location = 4) out vec4 gStructureBuffer;
layout (location = 5) out vec2 gVelocityBuffer;
layout (location = 6) out float gId;

// See Foundations of Game Engine Development: Volume 2 (The Structure Buffer)
vec4 calculateStructureOutput(float z) {
    // 0xFFFFE000 allows us to extract the upper 10 bits of precision. z - h then
    // Removes the upper 10 bits of precision and leaves us with at least 11 bits of precision.
    //
    // When h and z - h are later recombined, the result will have at least 21 of the original
    // 23 floating point mantissa.
    float h = uintBitsToFloat(floatBitsToUint(z) & 0xFFFFE000U);
    // See https://stackoverflow.com/questions/16365385/explanation-of-dfdx for an explanation of dFd(x|y)
    // They are effectively calculating change in depth between nearest neighbors
    return vec4(dFdx(z), dFdy(z), h, z - h);
}

// See https://learnopengl.com/Advanced-Lighting/Parallax-Mapping
// vec2 calculateDepthCoords(in Material material, in vec2 texCoords, in vec3 viewDir) {
//     float height = texture(material.depthMap, texCoords).r;
//     vec2 p = viewDir.xy * (height * 0.005);
//     return texCoords - p;
// }

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

const float maxReflectivity = 0.8;

void main() {
    Material material = materials[materialIndices[fsDrawID]];
    uint flags = material.flags;

    vec3 viewDir = normalize(viewPosition - fsPosition);

    //vec2 texCoords = bitwiseAndBool(flags, GPU_DEPTH_MAPPED) ? calculateDepthCoords(material, fsTexCoords, viewDir) : fsTexCoords;
    vec2 texCoords = fsTexCoords;

    vec4 baseColor = bool(fsDiffuseMapped) ? texture(material.diffuseMap, texCoords) : FLOAT4_TO_VEC4(material.diffuseColor);
    runAlphaTest(baseColor.a);

    vec3 normal = bool(fsNormalMapped) ? calculateNormal(material, texCoords) : (fsNormal + 1.0) * 0.5; // [-1, 1] -> [0, 1]

    float roughness = bool(fsRoughnessMapped) ? texture(material.roughnessMap, texCoords).r : material.metallicRoughness[1];
    float metallic = bool(fsMetallicMapped) ? texture(material.metallicMap, texCoords).r : material.metallicRoughness[0];
    //float roughness = material.metallicRoughness[1];
    //float metallic = material.metallicRoughness[0];
    // float roughness = material.metallicRoughness[1];
    // float metallic = material.metallicRoughness[0];
    // See https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/main/source/Renderer/shaders/material_info.glsl
    // See https://stackoverflow.com/questions/61140427/opengl-glsl-extract-metalroughness-map-to-metal-map-and-roughness-map
    vec2 metallicRoughness = bool(fsMetallicRoughnessMapped) ? texture(material.metallicRoughnessMap, texCoords).bg : vec2(metallic, roughness);
    metallic = metallicRoughness.x;
    roughness = metallicRoughness.y;

    vec3 emissive = bool(fsEmissiveMapped) ? emissiveTextureMultiplier * texture(material.emissiveMap, texCoords).rgb : FLOAT3_TO_VEC3(material.emissiveColor);

    // Coordinate space is set to world
    //gPosition = fsPosition;
    // gNormal = (normal + 1.0) * 0.5; // Converts back to [-1, 1]
    gNormal = normal;
    gAlbedo = vec4(baseColor.rgb, emissive.r);
    float reflectance = material.reflectance;
    //vec3 maxReflectivity = FLOAT3_TO_VEC3(material.maxReflectivity);
    reflectance = mix(reflectance, maxReflectivity, (1.0 - roughness) * 0.5);
    gReflectivityEmissive = vec2(mix(reflectance, maxReflectivity, metallic), emissive.g);
    //gBaseReflectivity = vec4(vec3(0.5), emissive.g);
    gRoughnessMetallicAmbient = vec3(roughness, metallic, emissive.b);
    //gStructureBuffer = calculateStructureOutput(fsViewSpacePos.z);
    gStructureBuffer = calculateStructureOutput(1.0 / gl_FragCoord.w);
    gVelocityBuffer = calculateVelocity(fsCurrentClipPos, fsPrevClipPos);
    gId = float(fsDrawID);

    // Small offset to help prevent z fighting in certain cases
    //gl_FragDepth = baseColor.a < 1.0 ? gl_FragCoord.z - ALPHA_DEPTH_OFFSET : gl_FragCoord.z;
}